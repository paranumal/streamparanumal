/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#define LIBP_MAJOR_VERSION 0
#define LIBP_MINOR_VERSION 5
#define LIBP_PATCH_VERSION 0
#define LIBP_VERSION       00500
#define LIBP_VERSION_STR   "0.5.0"

#include "core.hpp"
#include "comm.hpp"
#include "settings.hpp"

namespace libp {

void platformAddSettings(settings_t& settings);
void platformReportSettings(settings_t& settings);

namespace internal {

class iplatform_t {
 public:
  settings_t& settings;
  properties_t props;

  iplatform_t(settings_t& _settings):
    settings(_settings) {
  }
};

} //namespace internal


class platform_t {
private:
  std::shared_ptr<internal::iplatform_t> iplatform;

 public:
  comm_t comm;
  device_t device;

  platform_t()=default;

  platform_t(settings_t& settings) {

    iplatform = std::make_shared<internal::iplatform_t>(settings);

    comm = settings.comm;

    DeviceConfig();
    DeviceProperties();
  }

  platform_t(const platform_t &other)=default;
  platform_t& operator = (const platform_t &other)=default;

  bool isInitialized() const {
    return (iplatform!=nullptr);
  }

  void assertInitialized() const {
    LIBP_ABORT("Platform not initialized.",
               !isInitialized());
  }

  kernel_t buildKernel(std::string fileName, std::string kernelName,
                       properties_t& kernelInfo);

  template <typename T>
  deviceMemory<T> malloc(const size_t count,
                         const properties_t &prop = properties_t()) {
    assertInitialized();
    return deviceMemory<T>(device.malloc<T>(count, prop));
  }

  template <typename T>
  deviceMemory<T> malloc(const size_t count,
                         const memory<T> src,
                         const properties_t &prop = properties_t()) {
    assertInitialized();
    return deviceMemory<T>(device.malloc<T>(count, src.ptr(), prop));
  }

  template <typename T>
  deviceMemory<T> malloc(const memory<T> src,
                         const properties_t &prop = properties_t()) {
    assertInitialized();
    return deviceMemory<T>(device.malloc<T>(src.length(), src.ptr(), prop));
  }

  template <typename T>
  pinnedMemory<T> hostMalloc(const size_t count){
    assertInitialized();
    properties_t hostProp;
    hostProp["host"] = true;
    return pinnedMemory<T>(device.malloc<T>(count, nullptr, hostProp));
  }

  template <typename T>
  pinnedMemory<T> hostMalloc(const size_t count,
                             const memory<T> src){
    assertInitialized();
    properties_t hostProp;
    hostProp["host"] = true;
    return pinnedMemory<T>(device.malloc<T>(count, src.ptr(), hostProp));
  }

  template <typename T>
  pinnedMemory<T> hostMalloc(const memory<T> src){
    assertInitialized();
    properties_t hostProp;
    hostProp["host"] = true;
    return pinnedMemory<T>(device.malloc<T>(src.length(), src.ptr(), hostProp));
  }

  settings_t& settings() {
    assertInitialized();
    return iplatform->settings;
  }

  properties_t& props() {
    assertInitialized();
    return iplatform->props;
  }

  void finish() {
    device.finish();
  }

  const int rank() const {
    return comm.rank();
  }

  const int size() const {
    return comm.size();
  }

  int getDeviceCount(const std::string mode) {
    return occa::getDeviceCount(mode);
  }

  void setCacheDir(const std::string cacheDir) {
    occa::env::setOccaCacheDir(cacheDir);
  }

 private:
  void DeviceConfig();
  void DeviceProperties();

};

} //namespace libp

#endif
