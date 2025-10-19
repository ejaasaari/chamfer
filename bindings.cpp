#include "chamfer.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(chamfer, m) {
  m.doc() = "Chamfer distance-based nearest neighbor search";

  py::class_<Chamfer>(m, "Chamfer")
      .def(py::init([](py::array_t<float> train,
                       py::array_t<uint32_t> train_counts, int vec_dim) {
             py::buffer_info train_buf = train.request();
             py::buffer_info counts_buf = train_counts.request();

             if (train_buf.ndim != 1) {
               throw std::runtime_error("train must be a 1-dimensional array");
             }
             if (counts_buf.ndim != 1) {
               throw std::runtime_error(
                   "train_counts must be a 1-dimensional array");
             }

             const float *train_ptr = static_cast<float *>(train_buf.ptr);
             const uint32_t *counts_ptr =
                 static_cast<uint32_t *>(counts_buf.ptr);
             int num_train_points = counts_buf.shape[0];

             return new Chamfer(train_ptr, counts_ptr, vec_dim,
                                num_train_points);
           }),
           py::arg("train"), py::arg("train_counts"), py::arg("vec_dim"),
           R"doc(
            Initialize Chamfer index with training data.

            Parameters
            ----------
            train : numpy.ndarray
                Flattened array of training vectors (shape: [total_vectors * vec_dim])
            train_counts : numpy.ndarray
                Number of vectors per training point (shape: [num_train_points])
            vec_dim : int
                Dimensionality of each vector
        )doc")

      .def(
          "compute_chamfer_similarity",
          [](const Chamfer &self, py::array_t<float> q, uint32_t count,
             int train_point_idx) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 1) {
              throw std::runtime_error("query must be a 1-dimensional array");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            return self.compute_chamfer_similarity(q_ptr, count,
                                                   train_point_idx);
          },
          py::arg("q"), py::arg("count"), py::arg("train_point_idx"),
          R"doc(
                Compute Chamfer similarity between a query and a specific training point.

                Parameters
                ----------
                q : numpy.ndarray
                    Flattened query vectors (shape: [count * vec_dim])
                count : int
                    Number of vectors in the query
                train_point_idx : int
                    Index of the training point to compare against

                Returns
                -------
                float
                    Chamfer similarity score (higher is more similar)
            )doc")

      .def(
          "distance_to_indices",
          [](const Chamfer &self, py::array_t<float> q, uint32_t count,
             const std::vector<int> &indices) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 1) {
              throw std::runtime_error("query must be a 1-dimensional array");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            return self.distance_to_indices(q_ptr, count, indices);
          },
          py::arg("q"), py::arg("count"), py::arg("indices"),
          R"doc(
                Compute Chamfer distances to specific training point indices.

                Parameters
                ----------
                q : numpy.ndarray
                    Flattened query vectors (shape: [count * vec_dim])
                count : int
                    Number of vectors in the query
                indices : list of int
                    Indices of training points to compute distances to

                Returns
                -------
                list of float
                    Chamfer similarity scores in the same order as indices
            )doc")

      .def(
          "query_subset",
          [](const Chamfer &self, py::array_t<float> q, uint32_t count, int k,
             const std::vector<int> &indices) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 1) {
              throw std::runtime_error("query must be a 1-dimensional array");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            return self.query_subset(q_ptr, count, k, indices);
          },
          py::arg("q"), py::arg("count"), py::arg("k"), py::arg("indices"),
          R"doc(
                Find k nearest neighbors from a subset of training points.

                Parameters
                ----------
                q : numpy.ndarray
                    Flattened query vectors (shape: [count * vec_dim])
                count : int
                    Number of vectors in the query
                k : int
                    Number of nearest neighbors to return
                indices : list of int
                    Subset of training point indices to search

                Returns
                -------
                list of int
                    Indices of k nearest neighbors from the subset
            )doc")

      .def(
          "query",
          [](const Chamfer &self, py::array_t<float> q, uint32_t count, int k) {
            py::buffer_info q_buf = q.request();
            if (q_buf.ndim != 1) {
              throw std::runtime_error("query must be a 1-dimensional array");
            }
            const float *q_ptr = static_cast<float *>(q_buf.ptr);
            return self.query(q_ptr, count, k);
          },
          py::arg("q"), py::arg("count"), py::arg("k"),
          R"doc(
                Find k nearest neighbors from all training points.

                Parameters
                ----------
                q : numpy.ndarray
                    Flattened query vectors (shape: [count * vec_dim])
                count : int
                    Number of vectors in the query
                k : int
                    Number of nearest neighbors to return

                Returns
                -------
                list of int
                    Indices of k nearest neighbors
            )doc")

      .def(
          "batch_query",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<uint32_t> query_counts, int k) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error("queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const uint32_t *counts_ptr =
                static_cast<uint32_t *>(counts_buf.ptr);
            int num_queries = counts_buf.shape[0];

            return self.batch_query(queries_ptr, counts_ptr, num_queries, k);
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("k"),
          R"doc(
                Find k nearest neighbors for multiple queries at once.

                Parameters
                ----------
                queries : numpy.ndarray
                    Flattened array of all query vectors (shape: [total_query_vectors * vec_dim])
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])
                k : int
                    Number of nearest neighbors to return per query

                Returns
                -------
                list of list of int
                    For each query, indices of k nearest neighbors
            )doc")

      .def(
          "batch_query_subset",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<uint32_t> query_counts, int k,
             const std::vector<int> &indices) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error("queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const uint32_t *counts_ptr =
                static_cast<uint32_t *>(counts_buf.ptr);
            int num_queries = counts_buf.shape[0];

            return self.batch_query_subset(queries_ptr, counts_ptr, num_queries,
                                           k, indices);
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("k"),
          py::arg("indices"),
          R"doc(
                Find k nearest neighbors from a subset for multiple queries.

                Parameters
                ----------
                queries : numpy.ndarray
                    Flattened array of all query vectors (shape: [total_query_vectors * vec_dim])
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])
                k : int
                    Number of nearest neighbors to return per query
                indices : list of int
                    Subset of training point indices to search

                Returns
                -------
                list of list of int
                    For each query, indices of k nearest neighbors from the subset
            )doc")

      .def(
          "batch_distance_to_indices",
          [](const Chamfer &self, py::array_t<float> queries,
             py::array_t<uint32_t> query_counts,
             const std::vector<int> &indices) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();

            if (queries_buf.ndim != 1) {
              throw std::runtime_error("queries must be a 1-dimensional array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const uint32_t *counts_ptr =
                static_cast<uint32_t *>(counts_buf.ptr);
            int num_queries = counts_buf.shape[0];

            return self.batch_distance_to_indices(queries_ptr, counts_ptr,
                                                  num_queries, indices);
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("indices"),
          R"doc(
                Compute Chamfer distances to specific indices for multiple queries.

                Parameters
                ----------
                queries : numpy.ndarray
                    Flattened array of all query vectors (shape: [total_query_vectors * vec_dim])
                query_counts : numpy.ndarray
                    Number of vectors per query (shape: [num_queries])
                indices : list of int
                    Training point indices to compute distances to

                Returns
                -------
                list of list of float
                    For each query, Chamfer similarity scores to the specified indices
            )doc")

      .def(
          "batch_query_fixed",
          [](const Chamfer &self, py::array_t<float> queries, int k) {
            py::buffer_info queries_buf = queries.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];

            return self.batch_query_fixed(queries_ptr, num_queries,
                                          query_vec_count, k);
          },
          py::arg("queries"), py::arg("k"),
          R"doc(
                Find k nearest neighbors for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries have the same number of vectors.
                k : int
                    Number of nearest neighbors to return per query

                Returns
                -------
                list of list of int
                    For each query, indices of k nearest neighbors
            )doc")

      .def(
          "batch_query_subset_fixed",
          [](const Chamfer &self, py::array_t<float> queries, int k,
             const std::vector<int> &indices) {
            py::buffer_info queries_buf = queries.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];

            return self.batch_query_subset_fixed(queries_ptr, num_queries,
                                                 query_vec_count, k, indices);
          },
          py::arg("queries"), py::arg("k"), py::arg("indices"),
          R"doc(
                Find k nearest neighbors from a subset for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries have the same number of vectors.
                k : int
                    Number of nearest neighbors to return per query
                indices : list of int
                    Subset of training point indices to search

                Returns
                -------
                list of list of int
                    For each query, indices of k nearest neighbors from the subset
            )doc")

      .def(
          "batch_distance_to_indices_fixed",
          [](const Chamfer &self, py::array_t<float> queries,
             const std::vector<int> &indices) {
            py::buffer_info queries_buf = queries.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];

            return self.batch_distance_to_indices_fixed(
                queries_ptr, num_queries, query_vec_count, indices);
          },
          py::arg("queries"), py::arg("indices"),
          R"doc(
                Compute Chamfer distances to specific indices for multiple queries (fixed query size).

                Parameters
                ----------
                queries : numpy.ndarray
                    3D array of query vectors (shape: [num_queries, query_vec_count, vec_dim])
                    All queries have the same number of vectors.
                indices : list of int
                    Training point indices to compute distances to

                Returns
                -------
                list of list of float
                    For each query, Chamfer similarity scores to the specified indices
            )doc");
}
