// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Ported from cuCascade src/memory/topology_discovery.cpp (pinned at 1b0e7b6).
// Namespace and type names updated for cudf_streaming::prefetch::memory.
// Added NUMA node detail discovery (numa_topology_info) and hw_decompression_available.

#include <sirius/memory/topology_discovery.hpp>

#include <dlfcn.h>
#include <ifaddrs.h>
#include <nvml.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace cudf_streaming::prefetch::memory {

namespace {

struct NetworkDeviceWithTopology {
  std::string name;
  int numa_node;
  std::string pci_bus_id;
};

void report_nvml_error(nvmlReturn_t result, std::string const& context)
{
  std::cerr << "Warning: " << context << ": " << nvmlErrorString(result) << std::endl;
}

std::string read_file_content(std::string const& path)
{
  std::ifstream file(path);
  if (!file.is_open()) { return ""; }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();
  if (!content.empty() && content.back() == '\n') { content.pop_back(); }
  return content;
}

std::vector<int> parse_cpu_list(std::string const& cpulist)
{
  std::vector<int> cores;
  if (cpulist.empty()) { return cores; }

  std::istringstream iss(cpulist);
  std::string token;
  while (std::getline(iss, token, ',')) {
    size_t dash_pos = token.find('-');
    if (dash_pos != std::string::npos) {
      int start = std::stoi(token.substr(0, dash_pos));
      int end   = std::stoi(token.substr(dash_pos + 1));
      for (int i = start; i <= end; ++i) {
        cores.push_back(i);
      }
    } else {
      cores.push_back(std::stoi(token));
    }
  }
  return cores;
}

std::string normalize_pci_bus_id(std::string const& pci_bus_id)
{
  size_t colon_pos = pci_bus_id.find(':');
  if (colon_pos == std::string::npos) { return pci_bus_id; }
  std::string domain = pci_bus_id.substr(0, colon_pos);
  if (domain.length() > 4) { domain = domain.substr(domain.length() - 4); }

  std::string normalized_id = domain + pci_bus_id.substr(colon_pos);
  std::ranges::transform(normalized_id, normalized_id.begin(), ::tolower);

  return normalized_id;
}

std::string trim_copy(std::string const& input)
{
  size_t start = 0;
  while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
    ++start;
  }
  size_t end = input.size();
  while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
    --end;
  }
  return input.substr(start, end - start);
}

std::vector<std::string> split_csv(std::string const& input)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(input);
  while (std::getline(iss, token, ',')) {
    tokens.push_back(trim_copy(token));
  }
  return tokens;
}

bool is_numeric_token(std::string const& token)
{
  return !token.empty() &&
         std::ranges::all_of(token, [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::vector<size_t> resolve_visible_gpu_indices(
  std::vector<gpu_topology_info> const& nvml_gpus,
  std::unordered_map<std::string, size_t> const& index_by_pci,
  std::unordered_map<std::string, size_t> const& index_by_uuid)
{
  std::vector<size_t> indices;
  std::unordered_set<size_t> seen;

  char const* env_value = std::getenv("CUDA_VISIBLE_DEVICES");
  if (!env_value) {
    indices.reserve(nvml_gpus.size());
    for (size_t i = 0; i < nvml_gpus.size(); ++i) {
      indices.push_back(i);
    }
    return indices;
  }

  std::string env_str(env_value);
  auto tokens = split_csv(env_str);
  for (auto const& token : tokens) {
    if (token.empty()) { continue; }

    bool matched = false;
    if (is_numeric_token(token)) {
      size_t idx = 0;
      try {
        idx = static_cast<size_t>(std::stoul(token));
      } catch (std::exception const& e) {
        throw std::invalid_argument("Invalid numeric CUDA_VISIBLE_DEVICES entry: " + token);
      }
      if (idx < nvml_gpus.size()) {
        if (seen.insert(idx).second) { indices.push_back(idx); }
        matched = true;
      } else {
        throw std::invalid_argument("CUDA_VISIBLE_DEVICES entry " + token + " is out of range");
      }
    } else if (token.starts_with("GPU-") || token.starts_with("MIG-")) {
      auto uuid_it = index_by_uuid.find(token);
      if (uuid_it != index_by_uuid.end()) {
        if (seen.insert(uuid_it->second).second) { indices.push_back(uuid_it->second); }
        matched = true;
      }

      if (!matched) {
        nvmlDevice_t handle;
        if (nvmlDeviceGetHandleByUUID(token.c_str(), &handle) == NVML_SUCCESS) {
          unsigned int is_mig = 0;
          if (nvmlDeviceIsMigDeviceHandle(handle, &is_mig) == NVML_SUCCESS && is_mig) {
            nvmlDevice_t parent_handle;
            if (nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle, &parent_handle) ==
                NVML_SUCCESS) {
              handle = parent_handle;
            }
          }

          nvmlPciInfo_t pci_info;
          if (nvmlDeviceGetPciInfo_v3(handle, &pci_info) == NVML_SUCCESS) {
            std::string normalized = normalize_pci_bus_id(pci_info.busId);
            auto it                = index_by_pci.find(normalized);
            if (it != index_by_pci.end()) {
              if (seen.insert(it->second).second) { indices.push_back(it->second); }
              matched = true;
            }
          }
        }
      }
    }

    if (!matched) {
      std::cerr << "Warning: CUDA_VISIBLE_DEVICES entry '" << token
                << "' does not map to an NVML device" << std::endl;
    }
  }

  return indices;
}

int get_numa_node_from_nvml(nvmlDevice_t device)
{
  unsigned long nodeset = 0;
  if (nvmlDeviceGetMemoryAffinity(device, 1, &nodeset, NVML_AFFINITY_SCOPE_NODE) == NVML_SUCCESS &&
      nodeset != 0) {
    return std::countr_zero(nodeset);
  }
  return -1;
}

std::string get_cpu_affinity_from_sys(std::string const& pci_bus_id)
{
  std::string normalized_id = normalize_pci_bus_id(pci_bus_id);
  std::string path          = "/sys/bus/pci/devices/" + normalized_id + "/local_cpulist";
  return read_file_content(path);
}

std::string get_pci_bus_id_from_device(std::string const& device_path)
{
  fs::path device_link = fs::path(device_path) / "device";
  if (!fs::exists(device_link)) { return ""; }

  try {
    fs::path real_path = fs::canonical(device_link);
    return real_path.filename().string();
  } catch (...) {
    return "";
  }
}

int get_pci_bus_number(std::string const& pci_id)
{
  size_t first_colon = pci_id.find(':');
  if (first_colon == std::string::npos) { return -1; }

  size_t second_colon = pci_id.find(':', first_colon + 1);
  if (second_colon == std::string::npos) { return -1; }

  std::string bus_str = pci_id.substr(first_colon + 1, second_colon - first_colon - 1);
  try {
    return std::stoi(bus_str, nullptr, 16);
  } catch (...) {
    return -1;
  }
}

PciePathType get_pcie_path_type(std::string const& gpu_pci_id, std::string const& nic_pci_id)
{
  std::string gpu_norm = normalize_pci_bus_id(gpu_pci_id);
  std::string nic_norm = normalize_pci_bus_id(nic_pci_id);

  int gpu_numa = -1, nic_numa = -1;
  std::string gpu_numa_str = read_file_content("/sys/bus/pci/devices/" + gpu_norm + "/numa_node");
  std::string nic_numa_str = read_file_content("/sys/bus/pci/devices/" + nic_norm + "/numa_node");

  if (!gpu_numa_str.empty()) { gpu_numa = std::stoi(gpu_numa_str); }
  if (!nic_numa_str.empty()) { nic_numa = std::stoi(nic_numa_str); }

  if (gpu_numa != nic_numa && gpu_numa >= 0 && nic_numa >= 0) { return PciePathType::SYS; }

  int gpu_bus = get_pci_bus_number(gpu_pci_id);
  int nic_bus = get_pci_bus_number(nic_pci_id);

  if (gpu_bus < 0 || nic_bus < 0) { return PciePathType::PHB; }

  int bus_distance = std::abs(gpu_bus - nic_bus);

  if (bus_distance <= 2) {
    return PciePathType::PIX;
  } else if (bus_distance <= 10) {
    return PciePathType::PHB;
  } else {
    return PciePathType::NODE;
  }
}

bool has_active_port(std::string const& device_path)
{
  fs::path ports_dir = fs::path(device_path) / "ports";
  if (!fs::exists(ports_dir)) { return false; }

  try {
    for (auto const& port_entry : fs::directory_iterator(ports_dir)) {
      if (!port_entry.is_directory()) { continue; }

      std::string state = read_file_content((port_entry.path() / "state").string());
      if (state.find("ACTIVE") != std::string::npos) { return true; }
    }
  } catch (...) {
  }
  return false;
}

bool has_uverbs_device(std::string const& device_path)
{
  fs::path verbs_dir = fs::path(device_path) / "device" / "infiniband_verbs";
  if (!fs::exists(verbs_dir)) { return false; }

  try {
    for (auto const& entry : fs::directory_iterator(verbs_dir)) {
      if (!entry.is_directory()) { continue; }
      fs::path dev_node = fs::path("/dev/infiniband") / entry.path().filename();
      if (fs::exists(dev_node)) { return true; }
    }
  } catch (...) {
  }
  return false;
}

bool interface_has_ip(std::string const& iface_name)
{
  struct ifaddrs* ifa_list = nullptr;
  if (getifaddrs(&ifa_list) != 0) { return false; }

  bool found = false;
  for (struct ifaddrs* ifa = ifa_list; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) { continue; }
    int family = ifa->ifa_addr->sa_family;
    if ((family == AF_INET || family == AF_INET6) && iface_name == ifa->ifa_name) {
      found = true;
      break;
    }
  }

  freeifaddrs(ifa_list);
  return found;
}

bool has_net_interface_with_ip(std::string const& device_path)
{
  fs::path net_dir = fs::path(device_path) / "device" / "net";
  if (!fs::exists(net_dir)) { return false; }

  try {
    for (auto const& entry : fs::directory_iterator(net_dir)) {
      if (!entry.is_directory()) { continue; }
      if (interface_has_ip(entry.path().filename().string())) { return true; }
    }
  } catch (...) {
  }
  return false;
}

std::vector<NetworkDeviceWithTopology> discover_network_devices_with_topology(
  NetworkDeviceVerification verification)
{
  std::vector<NetworkDeviceWithTopology> devices;
  std::string ib_path = "/sys/class/infiniband";

  if (!fs::exists(ib_path)) { return devices; }

  try {
    for (auto const& entry : fs::directory_iterator(ib_path)) {
      if (!entry.is_directory()) { continue; }

      if (verification <= NetworkDeviceVerification::EXISTS_ACTIVE) {
        if (!has_active_port(entry.path().string())) { continue; }
        if (!has_uverbs_device(entry.path().string())) { continue; }
      }
      if (verification <= NetworkDeviceVerification::EXISTS_ACTIVE_IP) {
        if (!has_net_interface_with_ip(entry.path().string())) { continue; }
      }

      NetworkDeviceWithTopology dev;
      dev.name = entry.path().filename().string();

      std::string numa_path = entry.path().string() + "/device/numa_node";
      std::string numa_str  = read_file_content(numa_path);
      dev.numa_node         = numa_str.empty() ? -1 : std::stoi(numa_str);
      dev.pci_bus_id        = get_pci_bus_id_from_device(entry.path().string());

      devices.push_back(dev);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering network devices: " << e.what() << std::endl;
  }

  return devices;
}

std::vector<storage_device_info> discover_storage_devices_with_topology()
{
  std::vector<storage_device_info> devices;
  std::string nvme_path = "/sys/class/nvme";

  if (!fs::exists(nvme_path)) { return devices; }

  try {
    for (auto const& entry : fs::directory_iterator(nvme_path)) {
      if (!entry.is_directory()) { continue; }

      storage_device_info dev;
      dev.name = entry.path().filename().string();
      dev.type = StorageDriveType::NVME;

      std::string numa_path = entry.path().string() + "/device/numa_node";
      std::string numa_str  = read_file_content(numa_path);
      dev.numa_node         = numa_str.empty() ? -1 : std::stoi(numa_str);
      dev.pci_bus_id        = get_pci_bus_id_from_device(entry.path().string());

      devices.push_back(dev);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering NVMe devices: " << e.what() << std::endl;
  }

  return devices;
}

std::vector<std::string> map_network_devices_to_gpu(
  std::string const& gpu_pci_id,
  int gpu_numa_node,
  std::vector<network_device_info> const& network_devices)
{
  std::vector<std::string> mapped_devices;

  struct NicWithPath {
    std::string name;
    PciePathType path_type;
  };

  std::vector<NicWithPath> nics_with_paths;

  for (auto const& dev : network_devices) {
    if (dev.pci_bus_id.empty()) { continue; }

    NicWithPath nic;
    nic.name      = dev.name;
    nic.path_type = get_pcie_path_type(gpu_pci_id, dev.pci_bus_id);

    nics_with_paths.push_back(nic);
  }

  if (nics_with_paths.empty()) { return mapped_devices; }

  PciePathType best_path_type = PciePathType::SYS;
  for (auto const& nic : nics_with_paths) {
    if (nic.path_type < best_path_type) { best_path_type = nic.path_type; }
  }

  for (auto const& nic : nics_with_paths) {
    if (nic.path_type == best_path_type) { mapped_devices.push_back(nic.name); }
  }

  if (mapped_devices.empty()) {
    for (auto const& dev : network_devices) {
      if (dev.numa_node == gpu_numa_node) { mapped_devices.push_back(dev.name); }
    }
  }

  if (mapped_devices.empty() && !network_devices.empty()) {
    for (auto const& dev : network_devices) {
      mapped_devices.push_back(dev.name);
    }
  }

  return mapped_devices;
}

std::string get_hostname()
{
  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) == 0) { return std::string(hostname.data()); }
  return "";
}

// Parse a meminfo value from a line like "Node 0 MemTotal: 263907488 kB"
// Returns 0 if the line cannot be parsed.
std::size_t parse_meminfo_kb(std::string const& line)
{
  auto colon = line.rfind(':');
  if (colon == std::string::npos) { return 0; }
  std::string value_part = trim_copy(line.substr(colon + 1));
  // Strip trailing "kB"
  auto kb_pos = value_part.find(" kB");
  if (kb_pos != std::string::npos) { value_part = value_part.substr(0, kb_pos); }
  try {
    return static_cast<std::size_t>(std::stoull(value_part)) * 1024;
  } catch (...) {
    return 0;
  }
}

std::vector<numa_topology_info> discover_numa_nodes()
{
  std::vector<numa_topology_info> nodes;
  std::string numa_base = "/sys/devices/system/node";

  if (!fs::exists(numa_base)) { return nodes; }

  try {
    for (auto const& entry : fs::directory_iterator(numa_base)) {
      std::string name = entry.path().filename().string();
      if (!name.starts_with("node")) { continue; }
      if (!entry.is_directory()) { continue; }

      int node_id = -1;
      try {
        node_id = std::stoi(name.substr(4));
      } catch (...) {
        continue;
      }

      numa_topology_info info;
      info.id = node_id;

      // Check for CPUs
      std::string cpulist =
        read_file_content((entry.path() / "cpulist").string());
      info.has_cpus = !cpulist.empty() && cpulist != "\n";

      // Read memory info from meminfo
      std::ifstream meminfo_file((entry.path() / "meminfo").string());
      if (meminfo_file.is_open()) {
        std::string line;
        while (std::getline(meminfo_file, line)) {
          if (line.find("MemTotal") != std::string::npos) {
            info.memory_capacity = parse_meminfo_kb(line);
          } else if (line.find("MemFree") != std::string::npos) {
            info.free_memory = parse_meminfo_kb(line);
          }
        }
      }

      // A node is device memory if it has no CPUs but does have memory.
      // This covers GPU HBM nodes (e.g. Grace-Hopper, DGX Station).
      info.is_device_memory = !info.has_cpus && info.memory_capacity > 0;

      nodes.push_back(info);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering NUMA nodes: " << e.what() << std::endl;
  }

  std::sort(nodes.begin(), nodes.end(), [](auto const& a, auto const& b) {
    return a.id < b.id;
  });

  return nodes;
}

nvmlReturn_t initialize_nvml_for_current_process()
{
  static std::mutex mutex;
  static pid_t initialized_pid    = -1;
  static nvmlReturn_t init_result = NVML_ERROR_UNINITIALIZED;

  std::lock_guard<std::mutex> lock(mutex);
  pid_t const pid = getpid();
  if (initialized_pid != pid) {
    init_result     = nvmlInit_v2();
    initialized_pid = pid;
  }
  return init_result;
}

}  // namespace

bool topology_discovery::discover(NetworkDeviceVerification net_verification)
{
  system_topology_info topology;

  nvmlReturn_t result = initialize_nvml_for_current_process();
  if (result != NVML_SUCCESS) {
    report_nvml_error(result, "Failed to initialize NVML");
  }

  unsigned int device_count = 0;
  bool nvml_available       = false;
  if (result == NVML_SUCCESS) {
    result = nvmlDeviceGetCount_v2(&device_count);
    if (result != NVML_SUCCESS) {
      report_nvml_error(result, "Failed to get device count");
      device_count = 0;
    } else {
      nvml_available = true;
    }
  }

  std::vector<NetworkDeviceWithTopology> network_devices_with_topology =
    discover_network_devices_with_topology(net_verification);

  topology.hostname            = get_hostname();
  topology.numa_nodes          = discover_numa_nodes();
  topology.num_numa_nodes      = static_cast<int>(topology.numa_nodes.size());
  topology.num_gpus            = device_count;
  topology.num_network_devices = static_cast<int>(network_devices_with_topology.size());

  topology.network_devices.clear();
  for (auto const& dev : network_devices_with_topology) {
    network_device_info info;
    info.name       = dev.name;
    info.numa_node  = dev.numa_node;
    info.pci_bus_id = dev.pci_bus_id;
    topology.network_devices.push_back(info);
  }

  topology.storage_devices = discover_storage_devices_with_topology();

  topology.gpus.clear();

  std::vector<gpu_topology_info> nvml_gpus;
  std::unordered_map<std::string, size_t> nvml_index_by_pci;
  std::unordered_map<std::string, size_t> nvml_index_by_uuid;
  if (nvml_available) {
    auto emit_gpu = [&](nvmlDevice_t handle,
                        std::string const& parent_pci,
                        int parent_numa,
                        std::string const& parent_cpu_affinity,
                        std::vector<int> const& parent_cpu_cores,
                        std::vector<std::string> const& parent_nics) {
      gpu_topology_info gpu;

      std::array<char, NVML_DEVICE_NAME_BUFFER_SIZE> name{};
      nvmlReturn_t r = nvmlDeviceGetName(handle, name.data(), NVML_DEVICE_NAME_BUFFER_SIZE);
      gpu.name       = (r == NVML_SUCCESS) ? std::string(name.data()) : "Unknown";

      std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE> uuid{};
      r        = nvmlDeviceGetUUID(handle, uuid.data(), NVML_DEVICE_UUID_BUFFER_SIZE);
      gpu.uuid = (r == NVML_SUCCESS) ? std::string(uuid.data()) : "Unknown";

      gpu.pci_bus_id        = parent_pci;
      gpu.numa_node         = parent_numa;
      gpu.cpu_affinity_list = parent_cpu_affinity;
      gpu.cpu_cores         = parent_cpu_cores;
      if (parent_numa >= 0) { gpu.memory_binding.push_back(parent_numa); }
      gpu.network_devices = parent_nics;

      // hw_decompression_available: always false for now; no public NVML API exposes this.
      gpu.hw_decompression_available = false;

      nvml_gpus.push_back(std::move(gpu));
      if (!nvml_gpus.back().uuid.empty()) {
        nvml_index_by_uuid.emplace(nvml_gpus.back().uuid, nvml_gpus.size() - 1);
      }
    };

    for (unsigned int i = 0; i < device_count; ++i) {
      nvmlDevice_t device;
      result = nvmlDeviceGetHandleByIndex_v2(i, &device);
      if (result != NVML_SUCCESS) {
        report_nvml_error(result, "Failed to get handle for GPU " + std::to_string(i));
        continue;
      }

      nvmlPciInfo_t pci_info;
      result = nvmlDeviceGetPciInfo_v3(device, &pci_info);
      if (result != NVML_SUCCESS) {
        report_nvml_error(result, "Failed to get PCI info for GPU " + std::to_string(i));
        continue;
      }
      std::string parent_pci      = std::string(pci_info.busId);
      int parent_numa             = get_numa_node_from_nvml(device);
      std::string parent_cpu_aff  = get_cpu_affinity_from_sys(parent_pci);
      std::vector<int> parent_cpu = parse_cpu_list(parent_cpu_aff);
      std::vector<std::string> parent_nics =
        map_network_devices_to_gpu(parent_pci, parent_numa, topology.network_devices);

      unsigned int mig_current = NVML_DEVICE_MIG_DISABLE;
      unsigned int mig_pending = NVML_DEVICE_MIG_DISABLE;
      nvmlReturn_t mig_rc      = nvmlDeviceGetMigMode(device, &mig_current, &mig_pending);
      bool mig_enabled = (mig_rc == NVML_SUCCESS && mig_current == NVML_DEVICE_MIG_ENABLE);

      if (!mig_enabled) {
        emit_gpu(device, parent_pci, parent_numa, parent_cpu_aff, parent_cpu, parent_nics);
        nvml_index_by_pci.emplace(normalize_pci_bus_id(parent_pci), nvml_gpus.size() - 1);
        continue;
      }

      unsigned int max_mig = 0;
      nvmlReturn_t mc_rc   = nvmlDeviceGetMaxMigDeviceCount(device, &max_mig);
      if (mc_rc != NVML_SUCCESS) {
        report_nvml_error(
          mc_rc,
          "MIG enabled on GPU " + std::to_string(i) + " but failed to query MIG device count");
        max_mig = 0;
      }

      unsigned int emitted = 0;
      for (unsigned int mig_idx = 0; mig_idx < max_mig; ++mig_idx) {
        nvmlDevice_t mig_device;
        nvmlReturn_t r = nvmlDeviceGetMigDeviceHandleByIndex(device, mig_idx, &mig_device);
        if (r == NVML_ERROR_NOT_FOUND) { continue; }
        if (r != NVML_SUCCESS) {
          report_nvml_error(r,
                            "Failed to get MIG handle for GPU " + std::to_string(i) + " slot " +
                              std::to_string(mig_idx));
          continue;
        }
        emit_gpu(mig_device, parent_pci, parent_numa, parent_cpu_aff, parent_cpu, parent_nics);
        ++emitted;
      }

      if (emitted == 0) {
        std::cerr << "Warning: MIG enabled on GPU " << i << " but no MIG instances were enumerated"
                  << std::endl;
      }
    }
  }

  auto visible_indices =
    resolve_visible_gpu_indices(nvml_gpus, nvml_index_by_pci, nvml_index_by_uuid);
  topology.num_gpus = static_cast<unsigned int>(visible_indices.size());
  for (size_t visible_idx = 0; visible_idx < visible_indices.size(); ++visible_idx) {
    size_t nvml_idx = visible_indices[visible_idx];
    if (nvml_idx >= nvml_gpus.size()) { continue; }
    auto gpu = nvml_gpus[nvml_idx];
    gpu.id   = static_cast<unsigned int>(visible_idx);
    topology.gpus.push_back(std::move(gpu));
  }

  _topology = std::move(topology);
  return true;
}

}  // namespace cudf_streaming::prefetch::memory
