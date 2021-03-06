/*
The MIT License (MIT)
Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
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

#include "settings.hpp"

setting_t::setting_t(string shortkey_, string longkey_,
                     string name_, string val_,
                     string description_, vector<string> options_)
  : shortkey{shortkey_}, longkey{longkey_},
    name{name_}, val{val_},
    description{description_}, options{options_},
    check{0} {}

void setting_t::updateVal(const string newVal){
  if (!options.size()) {
    val = newVal;
  } else {
    for (size_t i=0;i<options.size();i++) {
      if (newVal==options[i]) {//valid
        val = newVal;
        return;
      }
    }
    stringstream ss;
    ss << "Value: \"" << newVal << "\" "
       << "not valid for setting " << name <<std::endl
       << "Possible values are: { ";
    for (size_t i=0;i<options.size()-1;i++) ss << options[i] << ", ";
    ss << options[options.size()-1] << " }" << std::endl;
    CEED_ABORT(ss.str());
  }
}

bool setting_t::compareVal(const string token) const {
  return !(val.find(token) == std::string::npos);
}

string setting_t::toString() const {
  stringstream ss;

  ss << "Name:     [" << name << "]" << std::endl;
  ss << "CL keys:  [" << shortkey << ", " << longkey << "]" << std::endl;
  ss << "Value:    " << val << std::endl;

  if (!description.empty())
    ss << "Description: " << description << std::endl;

  if (options.size()) {
    ss << "Possible values: { ";
    for (size_t i=0;i<options.size()-1;i++) ss << options[i] << ", ";
    ss << options[options.size()-1] << " }" << std::endl;
  }

  return ss.str();
}

string setting_t::PrintUsage() const {
  stringstream ss;

  ss << "Name:     [" << name << "]" << std::endl;
  ss << "CL keys:  [" << shortkey << ", " << longkey << "]" << std::endl;

  if (!description.empty())
    ss << "Description: " << description << std::endl;

  if (options.size()) {
    ss << "Possible values: { ";
    for (size_t i=0;i<options.size()-1;i++) ss << options[i] << ", ";
    ss << options[options.size()-1] << " }" << std::endl;
  }

  return ss.str();
}

std::ostream& operator<<(ostream& os, const setting_t& setting) {
  os << setting.toString();
  return os;
}

settings_t::settings_t(MPI_Comm& _comm):
  comm(_comm) {}

void settings_t::newSetting(const string shortkey, const string longkey,
                            const string name, const string val,
                            const string description,
                            const vector<string> options) {

  for(auto it = settings.begin(); it != settings.end(); ++it) {
    setting_t *setting = it->second;
    if (!setting->shortkey.compare(shortkey)) {
      stringstream ss;
      ss << "Setting with key: [" << shortkey << "] already exists.";
      CEED_ABORT(ss.str());
    }
    if (!setting->longkey.compare(longkey)) {
      stringstream ss;
      ss << "Setting with key: [" << longkey << "] already exists.";
      CEED_ABORT(ss.str());
    }
  }

  auto search = settings.find(name);
  if (search == settings.end()) {
    setting_t *S = new setting_t(shortkey, longkey, name, val, description, options);
    settings[name] = S;
    insertOrder.push_back(name);
  } else {
    stringstream ss;
    ss << "Setting with name: [" << name << "] already exists.";
    CEED_ABORT(ss.str());
  }
}

void settings_t::changeSetting(const string name, const string newVal) {
  auto search = settings.find(name);
  if (search != settings.end()) {
    setting_t* val = search->second;
    val->updateVal(newVal);
  } else {
    stringstream ss;
    ss << "Setting with name: [" << name << "] does not exist.";
    CEED_ABORT(ss.str());
  }
}

void settings_t::parseSettings(const int argc, char** argv) {

  for (int i = 1; i < argc; ) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
    {
       PrintUsage();
       MPI_Abort(MPI_COMM_WORLD,CEED_ERROR);
       return;
    }

    for(auto it = settings.begin(); it != settings.end(); ++it) {
      setting_t *setting = it->second;
      if (strcmp(argv[i], setting->shortkey.c_str()) == 0 ||
          strcmp(argv[i], setting->longkey.c_str()) == 0) {
        if (setting->check!=0) {
          stringstream ss;
          ss << "Cannot set setting [" << setting->name << "] twice in run command.";
          CEED_ABORT(ss.str());
        } else {
          if (strcmp(argv[i], "-v") == 0 ||
              strcmp(argv[i], "--verbose") == 0) {
            changeSetting("VERBOSE", "TRUE");
            i++;
          } else {
            changeSetting(setting->name, string(argv[i+1]));
            i+=2;
          }
          setting->check=1;
          break;
        }
      }
    }
  }
}

string settings_t::getSetting(const string name) const {
  auto search = settings.find(name);
  if (search != settings.end()) {
    setting_t* val = search->second;
    return val->getVal<string>();
  } else {
    stringstream ss;
    ss << "Unable to find setting: [" << name << "]";
    CEED_ABORT(ss.str());
    return string();
  }
}

bool settings_t::compareSetting(const string name, const string token) const {
  auto search = settings.find(name);
  if (search != settings.end()) {
    setting_t* val = search->second;
    return val->compareVal(token);
  } else {
    stringstream ss;
    ss << "Unable to find setting: [" << name.c_str() << "]";
    CEED_ABORT(ss.str());
    return false;
  }
}

void settings_t::report() {
  std::cout << "Settings:\n\n";
  for (size_t i = 0; i < insertOrder.size(); ++i) {
    const string &s = insertOrder[i];
    setting_t* val = settings[s];
    std::cout << *val << std::endl;
  }
}

void settings_t::reportSetting(const string name) const {
  auto search = settings.find(name);
  if (search != settings.end()) {
    setting_t* val = search->second;
    std::cout << *val << std::endl;
  } else {
    stringstream ss;
    ss << "Unable to find setting: [" << name.c_str() << "]";
    CEED_ABORT(ss.str());
  }
}

void settings_t::PrintUsage() {
  std::cout << "Usage:\n\n";
  for (size_t i = 0; i < insertOrder.size(); ++i) {
    const string &s = insertOrder[i];
    setting_t* val = settings[s];
    std::cout << val->PrintUsage() << std::endl;
  }

  std::cout << "Name:     [HELP]" << std::endl;
  std::cout << "CL keys:  [-h, --help]" << std::endl;
  std::cout << "Description: Print this help message" << std::endl;
}

settings_t::~settings_t() {
  for(auto it = settings.begin(); it != settings.end(); ++it)
    delete it->second;
}