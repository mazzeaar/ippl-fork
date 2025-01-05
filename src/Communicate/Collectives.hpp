#include "Communicate/DataTypes.h"

#include "Communicate/Communicator.h"
#include "Communicate/Operations.h"

namespace ippl {
    namespace mpi {
        template <typename T>
        void Communicator::sendrecv(const T* send_data, int send_count, int destination,
                                    int send_tag, T* recv_data, int recv_count, int source,
                                    int recv_tag) {
            MPI_Datatype type = get_mpi_datatype<T>(*send_data);
            MPI_Sendrecv(const_cast<T*>(send_data), send_count, type, destination, send_tag,
                         recv_data, recv_count, type, source, recv_tag, *comm_m, MPI_STATUS_IGNORE);
        }

        template <typename T>
        void Communicator::alltoall(const T* input, T* output, int count) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);
            MPI_Alltoall(const_cast<T*>(input), count, type, output, count, type, *comm_m);
        }

        template <typename T>
        void Communicator::alltoallv(const T* input, const int* send_counts,
                                     const int* send_displacements, T* output,
                                     const int* recv_counts, const int* recv_displacements) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);
            MPI_Alltoallv(const_cast<T*>(input), send_counts, send_displacements, type, output,
                          recv_counts, recv_displacements, type, *comm_m);
        }

        template <typename T>
        void Communicator::gather(const T* input, T* output, int count, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Gather(const_cast<T*>(input), count, type, output, count, type, root, *comm_m);
        }

        template <typename T>
        void Communicator::allgatherv(const T* input, int send_count, T* output,
                                      const int* recv_counts, const int* displacements) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Allgatherv(const_cast<T*>(input), send_count, type, output, recv_counts,
                           displacements, type, *comm_m);
        }

        template <typename T>
        void Communicator::allgather(const T* input, T* output, int count) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Allgather(const_cast<T*>(input), count, type, output, count, type, *comm_m);
        }

        template <typename T>
        void Communicator::gatherv(const T* input, T* output, int send_count,
                                   const int* recv_counts, const int* displacements, int root) {
            MPI_Datatype type = get_mpi_datatype<T>(*input);

            MPI_Gatherv(const_cast<T*>(input), send_count, type, output, recv_counts, displacements,
                        type, root, *comm_m);
        }

            template <typename T>
            void Communicator::scatter(const T* input, T* output, int count, int root) {
                MPI_Datatype type = get_mpi_datatype<T>(*input);

                MPI_Scatter(const_cast<T*>(input), count, type, output, count, type, root, *comm_m);
            }

            template <typename T, class Op>
            void Communicator::reduce(const T* input, T* output, int count, Op, int root) {
                MPI_Datatype type = get_mpi_datatype<T>(*input);

                MPI_Op mpiOp = get_mpi_op<Op, T>();

                MPI_Reduce(const_cast<T*>(input), output, count, type, mpiOp, root, *comm_m);
            }

            template <typename T, class Op>
            void Communicator::reduce(const T& input, T& output, int count, Op op, int root) {
                reduce(&input, &output, count, op, root);
            }

            template <typename T, class Op>
            void Communicator::allreduce(const T* input, T* output, int count, Op) {
                MPI_Datatype type = get_mpi_datatype<T>(*input);

                MPI_Op mpiOp = get_mpi_op<Op, T>();

                MPI_Allreduce(const_cast<T*>(input), output, count, type, mpiOp, *comm_m);
            }

            template <typename T, class Op>
            void Communicator::allreduce(const T& input, T& output, int count, Op op) {
                allreduce(&input, &output, count, op);
            }

            template <typename T, class Op>
            void Communicator::allreduce(T * inout, int count, Op) {
                MPI_Datatype type = get_mpi_datatype<T>(*inout);

                MPI_Op mpiOp = get_mpi_op<Op, T>();

                MPI_Allreduce(MPI_IN_PLACE, inout, count, type, mpiOp, *comm_m);
            }

            template <typename T, class Op>
            void Communicator::allreduce(T & inout, int count, Op op) {
                allreduce(&inout, count, op);
            }

            template <typename T, class Op>
            void Communicator::scan(T * input, T * output, int count, Op) {
                MPI_Datatype type = get_mpi_datatype<T>(*input);

                MPI_Op mpiOp = get_mpi_op<Op, T>();

                MPI_Scan(input, output, count, type, mpiOp, *comm_m);
            }

            template <typename T>
            void Communicator::broadcast(T * data, int count, int root) {
                MPI_Datatype type = get_mpi_datatype<T>(*data);

                MPI_Bcast(data, count, type, root, *comm_m);
            }
        }  // namespace mpi
    }  // namespace ippl
