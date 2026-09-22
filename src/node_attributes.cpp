/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "spark_dsg/node_attributes.h"

#include "spark_dsg/printing.h"
#include "spark_dsg/serialization/attribute_serialization.h"
#include "spark_dsg/serialization/binary_conversions.h"
#include "spark_dsg/serialization/json_conversions.h"
#include "spark_dsg/serialization/versioning.h"

namespace spark_dsg {

template <typename Scalar>
Scalar cosineSimilarity(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& a,
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& b,
                        const Scalar eps = static_cast<Scalar>(1e-6)) {
  const auto res = a.dot(b) / (a.norm() * b.norm() + eps);
  return std::clamp(res, static_cast<Scalar>(-1.0), static_cast<Scalar>(1.0));
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> batchCosineSimilarity(
    const std::vector<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& feature_vectors,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& query_vector,
    const Scalar eps = static_cast<Scalar>(1e-6)) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> similarities(feature_vectors.size());
  for (size_t i = 0; i < feature_vectors.size(); ++i) {
    similarities(i) = cosineSimilarity(feature_vectors[i], query_vector, eps);
  }
  return similarities;
}

template <typename T>
std::string showIterable(const T& iterable, size_t max_length = 80) {
  std::stringstream ss;
  ss << "[";
  auto iter = iterable.begin();
  while (iter != iterable.end()) {
    ss << *iter;

    ++iter;
    if (iter != iterable.end()) {
      ss << ", ";
    }

    if (max_length && ss.str().size() >= max_length) {
      ss << "...";
      break;
    }
  }
  ss << "]";

  return ss.str();
}

template <typename Scalar>
std::string quatToString(const Eigen::Quaternion<Scalar>& q) {
  std::stringstream ss;
  ss << "{w: " << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "}";
  return ss.str();
}

template <typename Derived>
bool matricesEqual(const Eigen::DenseBase<Derived>& lhs,
                   const Eigen::DenseBase<Derived>& rhs) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    return false;
  }

  bool same = true;
  for (int r = 0; r < lhs.rows(); ++r) {
    for (int c = 0; c < lhs.cols(); ++c) {
      const auto lhs_nan = std::isnan(lhs(r, c));
      const auto rhs_nan = std::isnan(rhs(r, c));
      // if one value is nan, this still works
      same &= (lhs_nan && rhs_nan) || lhs(r, c) == rhs(r, c);
    }
  }

  return same;
}

bool operator==(const NearestVertexInfo& lhs, const NearestVertexInfo& rhs) {
  const auto block_equal = Eigen::Map<const Eigen::Vector3i>(lhs.block) ==
                           Eigen::Map<const Eigen::Vector3i>(rhs.block);
  const auto pos_equal = Eigen::Map<const Eigen::Vector3d>(lhs.voxel_pos) ==
                         Eigen::Map<const Eigen::Vector3d>(rhs.voxel_pos);
  return block_equal && pos_equal && lhs.vertex == rhs.vertex && lhs.label == rhs.label;
}

std::ostream& operator<<(std::ostream& out, const NodeAttributes& attrs) {
  return attrs.fill_ostream(out);
}

NodeAttributes::NodeAttributes() : NodeAttributes(Eigen::Vector3d::Zero()) {}

NodeAttributes::NodeAttributes(const Eigen::Vector3d& pos)
    : position(pos), last_update_time_ns(0), is_active(false), is_predicted(false) {}

NodeAttributes::Ptr NodeAttributes::clone() const {
  return std::make_unique<NodeAttributes>(*this);
}

void NodeAttributes::transform(const Eigen::Isometry3d& transform) {
  position = transform * position;
}

bool NodeAttributes::operator==(const NodeAttributes& other) const {
  return is_equal(other);
}

std::ostream& NodeAttributes::fill_ostream(std::ostream& out) const {
  auto format = getDefaultVectorFormat();
  out << "  - position: " << position.transpose().format(format) << "\n";
  out << "  - last update time: "
      << (last_update_time_ns == 0 ? "n/a" : std::to_string(last_update_time_ns))
      << "\n";
  out << std::boolalpha << "  - is_active: " << is_active << "\n";
  out << std::boolalpha << "  - is_predicted: " << is_predicted;
  return out;
}

void NodeAttributes::serialization_info() {
  serialization::field("position", position);
  serialization::field("last_update_time_ns", last_update_time_ns);
  serialization::field("is_active", is_active);
  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version < io::Version(1, 0, 4)) {
    io::warnOutdatedHeader(header);
  } else {
    serialization::field("is_predicted", is_predicted);
  }
}

void NodeAttributes::serialization_info() const {
  const_cast<NodeAttributes*>(this)->serialization_info();
}

bool NodeAttributes::is_equal(const NodeAttributes& other) const {
  return matricesEqual(position, other.position) &&
         last_update_time_ns == other.last_update_time_ns &&
         is_active == other.is_active && is_predicted == other.is_predicted;
}

SemanticNodeAttributes::SemanticNodeAttributes()
    : NodeAttributes(),
      name(""),
      semantic_label(NO_SEMANTIC_LABEL),
      semantic_feature(0, 0) {}

NodeAttributes::Ptr SemanticNodeAttributes::clone() const {
  return std::make_unique<SemanticNodeAttributes>(*this);
}

void SemanticNodeAttributes::transform(const Eigen::Isometry3d& transform) {
  NodeAttributes::transform(transform);
  bounding_box.transform(transform);
}

bool SemanticNodeAttributes::hasLabel() const {
  return semantic_label != NO_SEMANTIC_LABEL;
}

bool SemanticNodeAttributes::hasFeature() const {
  return semantic_feature.rows() * semantic_feature.cols() != 0;
}

std::ostream& SemanticNodeAttributes::fill_ostream(std::ostream& out) const {
  NodeAttributes::fill_ostream(out);
  out << "\n  - color: " << color << "\n"
      << "  - name: '" << name << "'\n"
      << "  - bounding box: " << bounding_box << "\n"
      << "  - label: " << std::to_string(semantic_label) << "\n"
      << "  - feature: [" << semantic_feature.rows() << " x " << semantic_feature.cols()
      << "]";
  return out;
}

void SemanticNodeAttributes::serialization_info() {
  NodeAttributes::serialization_info();
  serialization::field("name", name);
  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version <= io::Version(1, 0, 2)) {
    io::warnOutdatedHeader(header);

    Eigen::Matrix<uint8_t, 3, 1> color_uint8;
    serialization::field("color", color_uint8);
    color = Color(color_uint8[0], color_uint8[1], color_uint8[2]);
  } else {
    serialization::field("color", color);
  }

  serialization::field("bounding_box", bounding_box);
  serialization::field("semantic_label", semantic_label);
  if (header.version <= io::Version(1, 0, 4)) {
    io::warnOutdatedHeader(header);

    Eigen::MatrixXd feature;
    serialization::field("semantic_feature", feature);
    semantic_feature = feature.cast<float>();
  } else {
    serialization::field("semantic_feature", semantic_feature);
  }
  if (header.version >= io::Version(1, 0, 4)) {
    serialization::field("label_weights", label_weights);
  }
}

bool SemanticNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const SemanticNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!NodeAttributes::is_equal(other)) {
    return false;
  }

  return name == derived->name && color == derived->color &&
         bounding_box == derived->bounding_box &&
         semantic_label == derived->semantic_label &&
         semantic_feature == derived->semantic_feature;
}

ObjectNodeAttributes::ObjectNodeAttributes()
    : SemanticNodeAttributes(),
      registered(false),
      world_R_object(Eigen::Quaterniond::Identity()) {}

NodeAttributes::Ptr ObjectNodeAttributes::clone() const {
  return std::make_unique<ObjectNodeAttributes>(*this);
}

void ObjectNodeAttributes::transform(const Eigen::Isometry3d& transform) {
  SemanticNodeAttributes::transform(transform);
  world_R_object =
      Eigen::Quaterniond(transform.linear() * world_R_object.toRotationMatrix());
}

double ObjectNodeAttributes::featureDistance(const Eigen::VectorXf& other) const {
  return static_cast<double>(cosineSimilarity(feature, other));
}

std::ostream& ObjectNodeAttributes::fill_ostream(std::ostream& out) const {
  SemanticNodeAttributes::fill_ostream(out);
  auto format = getDefaultVectorFormat();
  out << "\n  - mesh_connections: " << showIterable(mesh_connections);
  out << "\n  - registered?: " << (registered ? "yes" : "no");
  out << "\n  - world_R_object: " << quatToString(world_R_object);
  out << "\n  - num_observations: " << num_observations;
  out << "\n  - feature: [" << feature.transpose().format(format) << "]";
  return out;
}

void ObjectNodeAttributes::serialization_info() {
  SemanticNodeAttributes::serialization_info();
  serialization::field("mesh_connections", mesh_connections);
  serialization::field("registered", registered);
  serialization::field("world_R_object", world_R_object);
  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version <= io::Version(1, 0, 4)) {
    io::warnOutdatedHeader(header);

    Eigen::VectorXd sfeature;
    serialization::field("feature", sfeature);
    feature = sfeature.cast<float>();
  } else {
    serialization::field("feature", feature);
  }
  serialization::field("num_observations", num_observations);
}

template <typename Scalar>
bool quaternionsEqual(const Eigen::Quaternion<Scalar>& lhs,
                      const Eigen::Quaternion<Scalar>& rhs) {
  return lhs.w() == rhs.w() && lhs.x() == rhs.x() && lhs.y() == rhs.y() &&
         lhs.z() == rhs.z();
}

bool ObjectNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const ObjectNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!SemanticNodeAttributes::is_equal(other)) {
    return false;
  }

  return mesh_connections == derived->mesh_connections &&
         registered == derived->registered &&
         quaternionsEqual(world_R_object, derived->world_R_object);
}

GlobalFrontierNodeAttributes::GlobalFrontierNodeAttributes() : NodeAttributes() {}

NodeAttributes::Ptr GlobalFrontierNodeAttributes::clone() const {
  return std::make_unique<GlobalFrontierNodeAttributes>(*this);
}

double GlobalFrontierNodeAttributes::featureDistance(
    const Eigen::VectorXf& other) const {
  return static_cast<double>(cosineSimilarity(semantic_feature, other));
}

std::ostream& GlobalFrontierNodeAttributes::fill_ostream(std::ostream& out) const {
  NodeAttributes::fill_ostream(out);
  auto format = getDefaultVectorFormat();
  out << "\n  - connected_objects: " << showIterable(connected_objects);
  out << "\n  - connected_nav: " << connected_nav;
  out << "\n  - nav_layer: " << nav_layer;
  out << "\n  - semantic_feature: [" << semantic_feature.transpose().format(format)
      << "]";
  out << "\n  - direction: [" << direction.transpose().format(format) << "]";
  out << "\n  - use_nav_as_centroid: " << std::boolalpha << use_nav_as_centroid;
  out << "\n  - features: [" << features.size() << "]";
  out << "\n  - feature_points: [" << feature_points.size() << "]";
  out << "\n  - frontier_points: [" << frontier_points.size() << "]";
  out << "\n  - row_indices: " << showIterable(row_indices);
  out << "\n  - col_indices: " << showIterable(col_indices);
  return out;
}

void GlobalFrontierNodeAttributes::serialization_info() {
  NodeAttributes::serialization_info();
  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version <= io::Version(1, 0, 4)) {
    io::warnOutdatedHeader(header);

    Eigen::VectorXd sfeature;
    serialization::field("semantic_feature", sfeature);
    semantic_feature = sfeature.cast<float>();
  } else {
    serialization::field("semantic_feature", semantic_feature);
  }
  serialization::field("connected_objects", connected_objects);
  serialization::field("connected_nav", connected_nav);
  serialization::field("nav_layer", nav_layer);
  serialization::field("direction", direction);
  serialization::field("use_nav_as_centroid", use_nav_as_centroid);
  serialization::field("features", features);
  serialization::field("feature_points", feature_points);
  serialization::field("frontier_points", frontier_points);
  serialization::field("row_indices", row_indices);
  serialization::field("col_indices", col_indices);
}

bool GlobalFrontierNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const GlobalFrontierNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!NodeAttributes::is_equal(other)) {
    return false;
  }

  return semantic_class_labels == derived->semantic_class_labels &&
         connected_objects == derived->connected_objects &&
         connected_nav == derived->connected_nav && nav_layer == derived->nav_layer &&
         use_nav_as_centroid == derived->use_nav_as_centroid &&
         features == derived->features && feature_points == derived->feature_points &&
         frontier_points == derived->frontier_points &&
         row_indices == derived->row_indices && col_indices == derived->col_indices &&
         matricesEqual(semantic_feature, derived->semantic_feature) &&
         matricesEqual(direction, derived->direction);
}

RoomNodeAttributes::RoomNodeAttributes() : SemanticNodeAttributes() {}

NodeAttributes::Ptr RoomNodeAttributes::clone() const {
  return std::make_unique<RoomNodeAttributes>(*this);
}

double RoomNodeAttributes::featureDistance(const Eigen::VectorXf& other) const {
  const auto similarities = batchCosineSimilarity(features, other);
  return static_cast<double>(similarities.maxCoeff());
}

bool RoomNodeAttributes::validFeatures() const { return !features.empty(); }

std::ostream& RoomNodeAttributes::fill_ostream(std::ostream& out) const {
  SemanticNodeAttributes::fill_ostream(out);
  out << "\n  - features: [" << features.size() << "]";
  out << "\n  - num_observations: [" << num_observations.size() << "]";
  out << "\n  - label: " << label;
  return out;
}

void RoomNodeAttributes::serialization_info() {
  SemanticNodeAttributes::serialization_info();
  serialization::field("semantic_class_probabilities", semantic_class_probabilities);
  serialization::field("features", features);
  serialization::field("num_observations", num_observations);
  serialization::field("label", label);
}

bool RoomNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const RoomNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  return SemanticNodeAttributes::is_equal(other);
}

PlaceNodeAttributes::PlaceNodeAttributes() : PlaceNodeAttributes(0.0, 0) {}

PlaceNodeAttributes::PlaceNodeAttributes(double distance, unsigned int num_basis_points)
    : SemanticNodeAttributes(),
      distance(distance),
      num_basis_points(num_basis_points),
      frontier_scale(Eigen::Vector3d::Zero()),
      orientation(Eigen::Quaterniond::Identity()) {}

NodeAttributes::Ptr PlaceNodeAttributes::clone() const {
  return std::make_unique<PlaceNodeAttributes>(*this);
}

std::ostream& PlaceNodeAttributes::fill_ostream(std::ostream& out) const {
  SemanticNodeAttributes::fill_ostream(out);
  out << "\n  - distance: " << distance;
  out << "\n  - num basis points: " << num_basis_points;
  out << std::boolalpha << "\n  - real place: " << real_place;
  out << std::boolalpha << "\n  - need cleanup: " << need_cleanup;
  out << std::boolalpha << "\n  - active frontier: " << active_frontier;
  out << std::boolalpha << "\n  - anti frontier: " << active_frontier;
  out << "\n  - num frontier voxels: " << num_frontier_voxels;
  return out;
}

void PlaceNodeAttributes::serialization_info() {
  SemanticNodeAttributes::serialization_info();
  serialization::field("distance", distance);
  serialization::field("num_basis_points", num_basis_points);
  serialization::field("voxblox_mesh_connections", voxblox_mesh_connections);
  serialization::field("pcl_mesh_connections", pcl_mesh_connections);
  serialization::field("mesh_vertex_labels", mesh_vertex_labels);
  serialization::field("deformation_connections", deformation_connections);
  serialization::field("real_place", real_place);
  serialization::field("active_frontier", active_frontier);
  serialization::field("frontier_scale", frontier_scale);
  serialization::field("orientation", orientation);
  serialization::field("need_cleanup", need_cleanup);
  serialization::field("num_frontier_voxels", num_frontier_voxels);
  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version < io::Version(1, 1, 3)) {
    io::warnOutdatedHeader(header);
  } else {
    serialization::field("anti_frontier", anti_frontier);
  }
}

bool PlaceNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const PlaceNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!SemanticNodeAttributes::is_equal(other)) {
    return false;
  }

  return distance == derived->distance &&
         num_basis_points == derived->num_basis_points &&
         voxblox_mesh_connections == derived->voxblox_mesh_connections &&
         pcl_mesh_connections == derived->pcl_mesh_connections &&
         mesh_vertex_labels == derived->mesh_vertex_labels &&
         deformation_connections == derived->deformation_connections &&
         real_place == derived->real_place &&
         active_frontier == derived->active_frontier &&
         anti_frontier == derived->anti_frontier &&
         frontier_scale == derived->frontier_scale &&
         quaternionsEqual(orientation, derived->orientation) &&
         need_cleanup == derived->need_cleanup &&
         num_frontier_voxels == derived->num_frontier_voxels;
}

Place2dNodeAttributes::Place2dNodeAttributes()
    : Place2dNodeAttributes(std::vector<Eigen::Vector3d>()) {}

Place2dNodeAttributes::Place2dNodeAttributes(std::vector<Eigen::Vector3d> boundary)
    : SemanticNodeAttributes(),
      boundary(boundary),
      pcl_min_index(0),
      pcl_max_index(0),
      need_finish_merge(false),
      need_cleanup_splitting(false),
      has_active_mesh_indices(false) {}

NodeAttributes::Ptr Place2dNodeAttributes::clone() const {
  return std::make_unique<Place2dNodeAttributes>(*this);
}

std::ostream& Place2dNodeAttributes::fill_ostream(std::ostream& out) const {
  SemanticNodeAttributes::fill_ostream(out);
  out << "\n  - boundary.size(): " << boundary.size();
  return out;
}

void Place2dNodeAttributes::serialization_info() {
  SemanticNodeAttributes::serialization_info();
  serialization::field("boundary", boundary);
  serialization::field("ellipse_centroid", ellipse_centroid);
  serialization::field("ellipse_matrix_compress", ellipse_matrix_compress);
  serialization::field("ellipse_matrix_expand", ellipse_matrix_expand);
  serialization::field("pcl_boundary_connections", pcl_boundary_connections);
  serialization::field("voxblox_mesh_connections", voxblox_mesh_connections);
  serialization::field("pcl_mesh_connections", pcl_mesh_connections);
  serialization::field("mesh_vertex_labels", mesh_vertex_labels);
  serialization::field("deformation_connections", deformation_connections);
  serialization::field("need_cleanup_splitting", need_cleanup_splitting);
  serialization::field("has_active_mesh_indices", has_active_mesh_indices);
}

bool Place2dNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const Place2dNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!SemanticNodeAttributes::is_equal(other)) {
    return false;
  }

  return boundary == derived->boundary &&
         ellipse_centroid == derived->ellipse_centroid &&
         ellipse_matrix_compress == derived->ellipse_matrix_compress &&
         ellipse_matrix_expand == derived->ellipse_matrix_expand &&
         pcl_boundary_connections == derived->pcl_boundary_connections &&
         voxblox_mesh_connections == derived->voxblox_mesh_connections &&
         pcl_mesh_connections == derived->pcl_mesh_connections &&
         mesh_vertex_labels == derived->mesh_vertex_labels &&
         deformation_connections == derived->deformation_connections &&
         need_cleanup_splitting == derived->need_cleanup_splitting &&
         has_active_mesh_indices == derived->has_active_mesh_indices;
}

AgentNodeAttributes::AgentNodeAttributes() : NodeAttributes(), timestamp(0) {}

AgentNodeAttributes::AgentNodeAttributes(std::chrono::nanoseconds timestamp,
                                         const Eigen::Quaterniond& world_R_body,
                                         const Eigen::Vector3d& world_P_body,
                                         NodeId external_key)
    : NodeAttributes(world_P_body),
      timestamp(timestamp),
      world_R_body(world_R_body),
      external_key(external_key) {}

NodeAttributes::Ptr AgentNodeAttributes::clone() const {
  return std::make_unique<AgentNodeAttributes>(*this);
}

void AgentNodeAttributes::transform(const Eigen::Isometry3d& transform) {
  NodeAttributes::transform(transform);
  world_R_body = transform.linear() * world_R_body;
}

float AgentNodeAttributes::featureDistance(const Eigen::VectorXf& other) const {
  return cosineSimilarity(image_feature, other);
}

std::ostream& AgentNodeAttributes::fill_ostream(std::ostream& out) const {
  NodeAttributes::fill_ostream(out);
  out << "\n  - orientation: " << quatToString(world_R_body);
  return out;
}

void AgentNodeAttributes::serialization_info() {
  NodeAttributes::serialization_info();

  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version < io::Version(1, 1, 0)) {
    io::warnOutdatedHeader(header);
  } else {
    serialization::field("timestamp", timestamp);
  }

  serialization::field("world_R_body", world_R_body);
  serialization::field("external_key", external_key);
  serialization::field("dbow_ids", dbow_ids);
  serialization::field("dbow_values", dbow_values);
  serialization::field("image_feature", image_feature);
  serialization::field("image", image);
}

bool AgentNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const AgentNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!NodeAttributes::is_equal(other)) {
    return false;
  }

  return timestamp == derived->timestamp &&
         quaternionsEqual(world_R_body, derived->world_R_body) &&
         external_key == derived->external_key && dbow_ids == derived->dbow_ids &&
         dbow_values == derived->dbow_values;
}

KhronosObjectAttributes::KhronosObjectAttributes() : mesh(true, false, false) {}

NodeAttributes::Ptr KhronosObjectAttributes::clone() const {
  return std::make_unique<KhronosObjectAttributes>(*this);
}

std::ostream& KhronosObjectAttributes::fill_ostream(std::ostream& out) const {
  SemanticNodeAttributes::fill_ostream(out);
  out << "\n  - first_observed_ns: ";
  for (uint64_t t : first_observed_ns) {
    out << t << " ";
  }
  out << "\n  - last_observed_ns: ";
  for (uint64_t t : last_observed_ns) {
    out << t << " ";
  }
  out << "\n  - mesh: " << mesh.numVertices() << " vertices, " << mesh.numFaces()
      << " faces";
  return out;
}

void KhronosObjectAttributes::serialization_info() {
  ObjectNodeAttributes::serialization_info();
  serialization::field("first_observed_ns", first_observed_ns);
  serialization::field("last_observed_ns", last_observed_ns);
  serialization::field("trajectory_positions", trajectory_positions);
  serialization::field("trajectory_timestamps", trajectory_timestamps);
  serialization::field("dynamic_object_points", dynamic_object_points);
  serialization::field("details", details);

  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version <= io::Version(1, 0, 1)) {
    io::warnOutdatedHeader(header);

    std::vector<float> xyz;
    serialization::field("vertices", xyz);
    std::vector<uint8_t> rgb;
    serialization::field("colors", rgb);
    std::vector<uint32_t> faces;
    serialization::field("faces", faces);
    const auto num_vertices = xyz.size() / 3;
    const auto num_colors = rgb.size() / 4;
    const auto num_faces = faces.size();
    if (num_vertices != num_colors || num_vertices != num_faces) {
      return;
    }

    mesh = Mesh(true, false, false, false);
    mesh.resizeVertices(num_vertices);
    for (size_t i = 0; i < num_vertices; ++i) {
      mesh.setPos(i, {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]});
      mesh.setColor(i, {rgb[4 * i], rgb[4 * i + 1], rgb[4 * i + 2], rgb[4 * i + 3]});
    }

    mesh.resizeFaces(num_faces);
    for (size_t i = 0; i < num_faces; ++i) {
      auto& face = mesh.face(i);
      for (size_t j = 0; j < 3; ++j) {
        face[j] = faces[3 * i + j];
      }
    }
  } else {
    serialization::field("mesh", mesh);
  }
}

bool KhronosObjectAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const KhronosObjectAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!ObjectNodeAttributes::is_equal(other)) {
    return false;
  }

  return first_observed_ns == derived->first_observed_ns &&
         last_observed_ns == derived->last_observed_ns && mesh == derived->mesh &&
         trajectory_positions == derived->trajectory_positions &&
         dynamic_object_points == derived->dynamic_object_points &&
         details == derived->details;
}

bool BoundaryInfo::operator==(const BoundaryInfo& other) const {
  return min == other.min && max == other.max && states == other.states;
}

NodeAttributes::Ptr TraversabilityNodeAttributes::clone() const {
  return std::make_unique<TraversabilityNodeAttributes>(*this);
}

std::ostream& TraversabilityNodeAttributes::fill_ostream(std::ostream& out) const {
  NodeAttributes::fill_ostream(out);
  out << "  - min: " << boundary.min.transpose() << "\n"
      << "  - max: " << boundary.max.transpose() << "\n"
      << "  - first_observed_ns: " << first_observed_ns << "\n"
      << "  - last_observed_ns: " << last_observed_ns << "\n"
      << "  - distance: " << distance << "\n";
  return out;
}

void TraversabilityNodeAttributes::serialization_info() {
  NodeAttributes::serialization_info();
  serialization::field("first_observed_ns", first_observed_ns);
  serialization::field("last_observed_ns", last_observed_ns);
  serialization::field("distance", distance);
  serialization::field("min", boundary.min);
  serialization::field("max", boundary.max);
  // Workaround for state serialization.
  for (size_t i = 0; i < 4; ++i) {
    std::vector<uint8_t> s;
    s.reserve(boundary.states[i].size());
    for (const auto& state : boundary.states[i]) {
      s.push_back(static_cast<uint8_t>(state));
    }
    serialization::field("states_" + std::to_string(i), s);
    boundary.states[i].clear();
    boundary.states[i].reserve(s.size());
    for (const auto& state : s) {
      boundary.states[i].push_back(static_cast<TraversabilityState>(state));
    }
  }

  const auto& header = io::GlobalInfo::loadedHeader();
  if (header.version < io::Version(1, 0, 4)) {
    io::warnOutdatedHeader(header);
    std::map<int, float> temp;
    if (header.version == io::Version(1, 1, 3)) {
      serialization::field("daaam_labels", temp);
    } else {
      serialization::field("cognition_labels", temp);
    }
    label_weights.clear();
    for (const auto& [label, weight] : temp) {
      label_weights[static_cast<Label>(label)] = weight;
    }
  }
}

bool TraversabilityNodeAttributes::is_equal(const NodeAttributes& other) const {
  const auto derived = dynamic_cast<const TraversabilityNodeAttributes*>(&other);
  if (!derived) {
    return false;
  }

  if (!NodeAttributes::is_equal(other)) {
    return false;
  }

  return boundary == derived->boundary && distance == derived->distance &&
         first_observed_ns == derived->first_observed_ns &&
         last_observed_ns == derived->last_observed_ns;
}

}  // namespace spark_dsg
