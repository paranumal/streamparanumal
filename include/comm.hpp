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

#ifndef LIBP_COMM_HPP
#define LIBP_COMM_HPP

#include <mpi.h>
#include "core.hpp"

namespace libp {

/*Generic data type*/
template<typename T>
struct mpiType {
  static MPI_Datatype getMpiType() {
    MPI_Datatype type;
    MPI_Type_contiguous(sizeof(T), MPI_CHAR, &type);
    MPI_Type_commit(&type);
    return type;
  }
  static void freeMpiType(MPI_Datatype type) {
    MPI_Type_free(&type);
  }
  static constexpr bool isMpiType() { return false; }
};

/*Pre-defined MPI datatypes*/
#define TYPE(T, MPI_T)                               \
template<> struct mpiType<T> {                       \
  static MPI_Datatype getMpiType() { return MPI_T; } \
  static void freeMpiType(MPI_Datatype type) { }     \
  static constexpr bool isMpiType() { return true; } \
}

TYPE(char,   MPI_CHAR);
TYPE(int,    MPI_INT);
TYPE(long long int, MPI_LONG_LONG_INT);
TYPE(float,  MPI_FLOAT);
TYPE(double, MPI_DOUBLE);
#undef TYPE

/*Communicator class*/
class comm_t {

 private:
  int _rank=0;
  int _size=0;

 public:
  MPI_Comm comm=MPI_COMM_NULL;
  comm_t() = default;
  comm_t(const comm_t &c) = default;
  comm_t& operator = (const comm_t &c)=default;

  /*Static MPI_Init and MPI_Finalize*/
  static void Init(int &argc, char** &argv);
  static void Finalize();

  /*Static handle to MPI_COMM_WORLD*/
  static const comm_t world();

  /*MPI_Comm_dup and MPI_Comm_delete*/
  comm_t Dup();
  void Free();
  void Split(const comm_t &c, const int color, const int key);

  /*Rank and size getters*/
  const int rank() const;
  const int size() const;

  using request_t = MPI_Request;

  /*Predefined ops*/
  using op_t = MPI_Op;
  static constexpr op_t Max  = MPI_MAX;
  static constexpr op_t Min  = MPI_MIN;
  static constexpr op_t Sum  = MPI_SUM;
  static constexpr op_t Prod = MPI_PROD;
  static constexpr op_t And  = MPI_LAND;
  static constexpr op_t Or   = MPI_LOR;
  static constexpr op_t Xor  = MPI_LXOR;

  /*libp::memory send*/
  template <typename T>
  void Send(const memory<T> &m,
            const int dest,
            const int count=-1,
            const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Send(m.ptr(), cnt, type, dest, tag, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory recv*/
  template <typename T>
  void Recv(const memory<T> &m,
            const int source,
            const int count=-1,
            const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Recv(m.ptr(), cnt, type, source, tag, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory non-blocking send*/
  template <typename T>
  request_t Isend(memory<T> &m,
                  const int dest,
                  const int count=-1,
                  const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    request_t request;
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Isend(m.ptr(), cnt, type, dest, tag, comm, &request);
    mpiType<T>::freeMpiType(type);
    return request;
  }

  /*libp::memory non-blocking recv*/
  template <typename T>
  request_t Irecv(memory<T> &m,
                  const int source,
                  const int count=-1,
                  const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    request_t request;
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Irecv(m.ptr(), cnt, type, source, tag, comm, &request);
    mpiType<T>::freeMpiType(type);
    return request;
  }

  /*libp::memory broadcast*/
  template <typename T>
  void Bcast(memory<T> &m,
             const int root,
             const int count=-1,
             const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Bcast(m.ptr(), cnt, type, root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*scalar broadcast*/
  template <typename T>
  void Bcast(T &val,
             const int root,
             const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Bcast(&val, 1, type, root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory reduce*/
  template <typename T>
  void Reduce(const memory<T> &snd,
                    memory<T> &rcv,
              const int root,
              const op_t op = Sum,
              const int count=-1,
              const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(snd.length()) : count;
    MPI_Reduce(snd.ptr(), rcv.ptr(), cnt, type, op, root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory in-place reduce*/
  template <typename T>
  void Reduce(memory<T> &m,
              const int root,
              const op_t op = Sum,
              const int count=-1,
              const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    if (_rank==root) {
      MPI_Reduce(MPI_IN_PLACE, m.ptr(), cnt, type, op, root, comm);
    } else {
      MPI_Reduce(m.ptr(), nullptr, cnt, type, op, root, comm);
    }
    mpiType<T>::freeMpiType(type);
  }

  /*scalar reduce*/
  template <typename T>
  void Reduce(const T &snd,
                    T &rcv,
              const int root,
              const op_t op = Sum,
              const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Reduce(&snd, &rcv, 1, type, op, root, comm);
    mpiType<T>::freeMpiType(type);
  }
  template <typename T>
  T Reduce(const T &val,
           const int root,
           const op_t op = Sum,
           const int tag=0) {
    T rcv=val;
    Reduce(val, rcv, root, op, tag);
    return rcv;
  }

  /*libp::memory allreduce*/
  template <typename T>
  void Allreduce(const memory<T> &snd,
                       memory<T> &rcv,
                 const op_t op = Sum,
                 const int count=-1,
                 const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(snd.length()) : count;
    MPI_Allreduce(snd.ptr(), rcv.ptr(), cnt, type, op, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory in-place allreduce*/
  template <typename T>
  void Allreduce(memory<T> &m,
                 const op_t op = Sum,
                 const int count=-1,
                 const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Allreduce(MPI_IN_PLACE, m.ptr(), cnt, type, op, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*scalar allreduce*/
  template <typename T>
  void Allreduce(const T &snd,
                       T &rcv,
                 const op_t op = Sum,
                 const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Allreduce(&snd, &rcv, 1, type, op, comm);
    mpiType<T>::freeMpiType(type);
  }
  template <typename T>
  T Allreduce(const T &snd,
              const op_t op = Sum,
              const int tag=0) {
    T rcv;
    Allreduce(snd, rcv, op, tag);
    return rcv;
  }

  /*libp::memory non-blocking allreduce*/
  template <typename T>
  request_t Iallreduce(const memory<T> &snd,
                             memory<T> &rcv,
                       const op_t op = Sum,
                       const int count=-1,
                       const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(snd.length()) : count;
    request_t request;
    MPI_Iallreduce(snd.ptr(), rcv.ptr(), cnt, type, op, comm, &request);
    mpiType<T>::freeMpiType(type);
    return request;
  }

  /*libp::memory non-blocking in-place allreduce*/
  template <typename T>
  request_t Iallreduce(memory<T> &m,
                       const int root,
                       const op_t op = Sum,
                       const int count=-1,
                       const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    request_t request;
    MPI_Iallreduce(MPI_IN_PLACE, m.ptr(), cnt, type, op, comm, &request);
    mpiType<T>::freeMpiType(type);
    return request;
  }

  /*scalar non-blocking allreduce*/
  template <typename T>
  request_t Iallreduce(const T &snd,
                             T &rcv,
                       const op_t op = Sum,
                       const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    request_t request;
    MPI_Iallreduce(&snd, &rcv, 1, type, op, comm, &request);
    mpiType<T>::freeMpiType(type);
    return request;
  }
  /*scalar non-blocking in-place allreduce*/
  template <typename T>
  request_t Iallreduce(T &val,
                       const op_t op = Sum,
                       const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    request_t request;
    MPI_Iallreduce(MPI_IN_PLACE, &val, 1, type, op, comm, &request);
    mpiType<T>::freeMpiType(type);
    return request;
  }

  /*libp::memory scan*/
  template <typename T>
  void Scan(const memory<T> &snd,
                  memory<T> &rcv,
            const op_t op = Sum,
            const int count=-1,
            const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(snd.length()) : count;
    MPI_Scan(snd.ptr(), rcv.ptr(), cnt, type, op, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory in-place scan*/
  template <typename T>
  void Scan(memory<T> &m,
            const op_t op = Sum,
            const int count=-1,
            const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(m.length()) : count;
    MPI_Scan(MPI_IN_PLACE, m.ptr(), cnt, type, op, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*scalar scan*/
  template <typename T>
  void Scan(const T &snd,
                  T &rcv,
            const op_t op = Sum,
            const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Scan(&snd, &rcv, 1, type, op, comm);
    mpiType<T>::freeMpiType(type);
  }
  template <typename T>
  T Scan(const T &snd,
         const op_t op = Sum,
         const int tag=0) {
    T rcv;
    Scan(snd, rcv, op, tag);
    return rcv;
  }

  /*libp::memory gather*/
  template <typename T>
  void Gather(const memory<T> &snd,
                    memory<T> &rcv,
              const int root,
              const int sendCount=-1,
              const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (sendCount==-1) ? static_cast<int>(snd.length()) : sendCount;
    MPI_Gather(snd.ptr(), cnt, type,
               rcv.ptr(), cnt, type, root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory gatherv*/
  template <typename T>
  void Gatherv(const memory<T> &snd,
               const int sendcount,
                     memory<T> &rcv,
               const memory<int> &recvCounts,
               const memory<int> &recvOffsets,
               const int root,
               const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Gatherv(snd.ptr(), sendcount, type,
                rcv.ptr(), recvCounts.ptr(), recvOffsets.ptr(), type,
                root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*scalar gather*/
  template <typename T>
  void Gather(const T &snd,
                    memory<T> &rcv,
              const int root,
              const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Gather(&snd,      1, type,
               rcv.ptr(), 1, type, root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory scatter*/
  template <typename T>
  void Scatter(const memory<T> &snd,
                     memory<T> &rcv,
               const int root,
               const int count=-1,
               const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (count==-1) ? static_cast<int>(rcv.length()) : count;
    MPI_Scatter(snd.ptr(), cnt, type,
                rcv.ptr(), cnt, type, root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory scatterv*/
  template <typename T>
  void Scatterv(const memory<T> &snd,
                const memory<int> &sendCounts,
                const memory<int> &sendOffsets,
                      memory<T> &rcv,
                const int recvcount,
                const int root,
                const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Scatterv(snd.ptr(), sendCounts.ptr(), sendOffsets.ptr(), type,
                 rcv.ptr(), recvcount, type,
                 root, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*scalar scatter*/
  template <typename T>
  T Scatter(const memory<T> &snd,
            const int root,
            const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    T rcv;
    MPI_Scatter(snd.ptr,   1, type,
                &rcv,      1, type, root, comm);
    mpiType<T>::freeMpiType(type);
    return rcv;
  }

  /*libp::memory allgather*/
  template <typename T>
  void Allgather(const memory<T> &snd,
                       memory<T> &rcv,
                 const int sendCount=-1,
                 const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    const int cnt = (sendCount==-1) ? static_cast<int>(snd.length()) : sendCount;
    MPI_Allgather(snd.ptr(), cnt, type,
                  rcv.ptr(), cnt, type, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory allgatherv*/
  template <typename T>
  void Allgatherv(const memory<T> &snd,
                  const int sendcount,
                        memory<T> &rcv,
                  const memory<int> &recvCounts,
                  const memory<int> &recvOffsets,
                  const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Allgatherv(snd.ptr(), sendcount, type,
                   rcv.ptr(), recvCounts.ptr(), recvOffsets.ptr(), type,
                   comm);
    mpiType<T>::freeMpiType(type);
  }

  /*scalar allgather*/
  template <typename T>
  void Allgather(const T &snd,
                       memory<T> &rcv,
                 const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Allgather(&snd,      1, type,
                  rcv.ptr(), 1, type, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory alltoall*/
  template <typename T>
  void Alltoall(const memory<T> &snd,
                      memory<T> &rcv,
                const int cnt,
                const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Alltoall(snd.ptr(), cnt, type,
                 rcv.ptr(), cnt, type, comm);
    mpiType<T>::freeMpiType(type);
  }

  /*libp::memory alltoallv*/
  template <typename T>
  void Allgatherv(const memory<T> &snd,
                  const memory<int> &sendCounts,
                  const memory<int> &sendOffsets,
                        memory<T> &rcv,
                  const memory<int> &recvCounts,
                  const memory<int> &recvOffsets,
                  const int tag=0) {
    MPI_Datatype type = mpiType<T>::getMpiType();
    MPI_Alltoallv(snd.ptr(), sendCounts.ptr(), sendOffsets.ptr(), type,
                  rcv.ptr(), recvCounts.ptr(), recvOffsets.ptr(), type,
                  comm);
    mpiType<T>::freeMpiType(type);
  }

  void Wait(request_t &request);
  void WaitAll(const int count, memory<request_t> &requests);
  void Barrier();
};

} //namespace libp

#endif
