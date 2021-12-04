// Copyright 2019 The MathWorks, Inc.
#include "MWCUFFTPlanManager.hpp"

#include "cufft.h"
#include <vector>
#include <map>

#ifdef MW_GPUCODER_RUNTIME_LOG
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ctime>
#endif

namespace gcutil {

#ifdef MW_GPUCODER_RUNTIME_LOG
void mwGpuCoderRuntimeLog(const std::string& msg) {
    const std::string LOG_FILE_NAME("gpucoder_runtime_log.txt");
    std::ofstream logFile;
    std::time_t currentTime = std::time(nullptr);
    logFile.open(LOG_FILE_NAME, std::ios::app);
    logFile << std::asctime(std::localtime(&currentTime));
    logFile << msg << std::endl;
    logFile.close();
}
#endif

class CUFFTPlanConfig {
  public:
    CUFFTPlanConfig(int rank,
                    int* n,
                    int* inembed,
                    int istride,
                    int idist,
                    int* onembed,
                    int ostride,
                    int odist,
                    cufftType type,
                    int batch)
        : m_rank(rank)
        , m_n(n, n + rank)
        , m_istride(istride)
        , m_idist(idist)
        , m_ostride(ostride)
        , m_odist(odist)
        , m_type(type)
        , m_batch(batch) {
        if (inembed != NULL) {
            m_inembed = std::vector<int>(inembed, inembed + rank);
        }
        if (onembed != NULL) {
            m_onembed = std::vector<int>(onembed, onembed + rank);
        }
    }

    bool operator<(const CUFFTPlanConfig& other) const {
        if (m_rank < other.m_rank) {
            return true;
        } else if (m_rank > other.m_rank) {
            return false;
        }

        if (m_n < other.m_n) {
            return true;
        } else if (m_n > other.m_n) {
            return false;
        }

        if (m_inembed < other.m_inembed) {
            return true;
        } else if (m_inembed > other.m_inembed) {
            return false;
        }

        if (m_istride < other.m_istride) {
            return true;
        } else if (m_istride > other.m_istride) {
            return false;
        }

        if (m_idist < other.m_idist) {
            return true;
        } else if (m_idist > other.m_idist) {
            return false;
        }

        if (m_onembed < other.m_onembed) {
            return true;
        } else if (m_onembed > other.m_onembed) {
            return false;
        }

        if (m_ostride < other.m_ostride) {
            return true;
        } else if (m_ostride > other.m_ostride) {
            return false;
        }

        if (m_odist < other.m_odist) {
            return true;
        } else if (m_odist > other.m_odist) {
            return false;
        }

        if (m_type < other.m_type) {
            return true;
        } else if (m_type > other.m_type) {
            return false;
        }

        if (m_batch < other.m_batch) {
            return true;
        } else if (m_batch > other.m_batch) {
            return false;
        }

        return false;
    }

  private:
    int m_rank;
    std::vector<int> m_n;
    std::vector<int> m_inembed;
    int m_istride;
    int m_idist;
    std::vector<int> m_onembed;
    int m_ostride;
    int m_odist;
    cufftType m_type;
    int m_batch;
};

class CUFFTPlanManager {
  public:
    typedef std::map<CUFFTPlanConfig, cufftHandle> CacheType;
    static CUFFTPlanManager& getInstance();

  public:
    void destroyPlans();
    cufftHandle acquirePlan(int rank,
                            int* n,
                            int* inembed,
                            int istride,
                            int idist,
                            int* onembed,
                            int ostride,
                            int odist,
                            cufftType type,
                            int batch);
    ~CUFFTPlanManager() {
        destroyPlans();
    }

  private:
    CUFFTPlanManager();
    CUFFTPlanManager(const CUFFTPlanManager&);
    CUFFTPlanManager& operator=(const CUFFTPlanManager&);

  private:
    CacheType m_configHandleMap;
    size_t m_workSize;
    void* m_workArea;
};
} // namespace gcutil

namespace gcutil {

CUFFTPlanManager::CUFFTPlanManager()
    : m_workSize(0)
    , m_workArea(NULL) {
}

CUFFTPlanManager& CUFFTPlanManager::getInstance() {
    static CUFFTPlanManager s_instance;
    return s_instance;
}

cufftHandle CUFFTPlanManager::acquirePlan(int rank,
                                          int* n,
                                          int* inembed,
                                          int istride,
                                          int idist,
                                          int* onembed,
                                          int ostride,
                                          int odist,
                                          cufftType type,
                                          int batch) {
    CUFFTPlanConfig config(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
    if (m_configHandleMap.find(config) != m_configHandleMap.end()) {
        cufftHandle handle = m_configHandleMap.at(config);
        cufftSetWorkArea(handle, m_workArea);
        return handle;
    } else {
        cufftHandle handle;
        cufftCreate(&handle);
        ;
        cufftSetAutoAllocation(handle, 0);
        size_t workSize;
        cufftMakePlanMany(handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type,
                          batch, &workSize);
        if (workSize > m_workSize) {
            m_workSize = workSize;
            cudaFree(m_workArea);
            cudaMalloc(&m_workArea, m_workSize);
        }
        cufftSetWorkArea(handle, m_workArea);
        m_configHandleMap[config] = handle;
#ifdef MW_GPUCODER_RUNTIME_LOG
        std::stringstream msgStream;
        msgStream << "Created cufftHandle : " << handle << ", workSize = " << m_workSize;
        mwGpuCoderRuntimeLog(msgStream.str());
#endif
        return handle;
    }
}

void CUFFTPlanManager::destroyPlans() {

    for (CUFFTPlanManager::CacheType::iterator t = m_configHandleMap.begin();
         t != m_configHandleMap.end(); t++) {
        cufftHandle handle = t->second;
        cufftDestroy(handle);
    }
    m_workSize = 0;
    cudaFree(m_workArea);
    m_configHandleMap.clear();
}

} // namespace gcutil

cufftHandle acquireCUFFTPlan(int nelem, cufftType type, int batch, int idist) {
    return gcutil::CUFFTPlanManager::getInstance().acquirePlan(1, &nelem, &idist, 1, idist, &nelem,
                                                               1, nelem, type, batch);
}
