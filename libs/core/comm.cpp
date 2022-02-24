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

#include "comm.hpp"

namespace libp {

/*Static MPI_Init and MPI_Finalize*/
void comm_t::Init(int &argc, char** &argv) { MPI_Init(&argc, &argv); }
void comm_t::Finalize() { MPI_Finalize(); }

/*Static handle to MPI_COMM_WORLD*/
const comm_t comm_t::world() {
  comm_t c;
  c.comm = MPI_COMM_WORLD;
  MPI_Comm_rank(c.comm, &(c._rank));
  MPI_Comm_size(c.comm, &(c._size));
  return c;
}

/*MPI_Comm_dup and MPI_Comm_free*/
comm_t comm_t::Dup() {
  comm_t c;
  MPI_Comm_dup(comm, &(c.comm));
  MPI_Comm_rank(c.comm, &(c._rank));
  MPI_Comm_size(c.comm, &(c._size));
  return c;
}
void comm_t::Free() {
  MPI_Comm_free(&comm);
  _rank=0;
  _size=0;
}
/*Split*/
void comm_t::Split(const comm_t &c, const int color, const int key) {
  MPI_Comm_split(c.comm, color, key, &comm);
  MPI_Comm_rank(comm, &_rank);
  MPI_Comm_size(comm, &_size);
}

/*Rank and size getters*/
const int comm_t::rank() const {
  return _rank;
}
const int comm_t::size() const {
  return _size;
}

void comm_t::Wait(request_t &request) {
  MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void comm_t::WaitAll(const int count, memory<request_t> &requests) {
  MPI_Waitall(count, requests.ptr(), MPI_STATUSES_IGNORE);
}

void comm_t::Barrier() {
  MPI_Barrier(comm);
}

} //namespace libp
