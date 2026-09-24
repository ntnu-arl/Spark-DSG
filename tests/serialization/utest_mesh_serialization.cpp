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
#include <gtest/gtest.h>

#include "spark_dsg/mesh.h"

namespace spark_dsg {

namespace {

void addTrackData(Mesh& mesh) {
  mesh.track_ids.push_back({{10u, 11u, Mesh::noTrack(), Mesh::noTrack()}});
  mesh.track_ids.push_back({{20u, 21u, 22u, Mesh::noTrack()}});
  mesh.track_ids.push_back({{30u, Mesh::noTrack(), Mesh::noTrack(), Mesh::noTrack()}});

  mesh.track_likelihoods.push_back({{1.0f, 0.5f, 0.0f, 0.0f}});
  mesh.track_likelihoods.push_back({{2.0f, 1.5f, 1.0f, 0.0f}});
  mesh.track_likelihoods.push_back({{3.0f, 0.0f, 0.0f, 0.0f}});
}

void addConfirmedTrackData(Mesh& mesh) {
  mesh.confirmed_track_ids.push_back({{10u, Mesh::noTrack(), Mesh::noTrack(), Mesh::noTrack()}});
  mesh.confirmed_track_ids.push_back({{20u, 21u, Mesh::noTrack(), Mesh::noTrack()}});
  mesh.confirmed_track_ids.push_back({{30u, 31u, 32u, Mesh::noTrack()}});

  mesh.confirmed_track_likelihoods.push_back({{0.75f, 0.0f, 0.0f, 0.0f}});
  mesh.confirmed_track_likelihoods.push_back({{1.25f, 0.5f, 0.0f, 0.0f}});
  mesh.confirmed_track_likelihoods.push_back({{2.0f, 1.0f, 0.25f, 0.0f}});
}

}  // namespace

TEST(MeshSerialization, MeshJson) {
  Mesh mesh(true, true, true, false, true, true);
  mesh.points.push_back(Eigen::Vector3f::Zero());
  mesh.points.push_back(Eigen::Vector3f::Zero());
  mesh.points.push_back(Eigen::Vector3f::Zero());
  mesh.colors.push_back({10, 20, 30, 255});
  mesh.labels.push_back(2);
  mesh.labels.push_back(8);
  mesh.stamps.push_back(0);
  mesh.stamps.push_back(10);
  mesh.stamps.push_back(20);
  mesh.faces.push_back({{1, 2, 3}});
  addTrackData(mesh);
  addConfirmedTrackData(mesh);

  const auto output = mesh.serializeToJson();
  auto result = Mesh::deserializeFromJson(output);

  ASSERT_TRUE(result);
  EXPECT_EQ(result->points.size(), 3u);
  EXPECT_EQ(result->colors.size(), 1u);
  EXPECT_EQ(result->labels.size(), 2u);
  EXPECT_EQ(result->stamps.size(), 3u);
  EXPECT_TRUE(result->has_track_data);
  EXPECT_TRUE(result->has_confirmed_track_data);
  EXPECT_EQ(result->track_ids.size(), 3u);
  EXPECT_EQ(result->track_likelihoods.size(), 3u);
  EXPECT_EQ(result->confirmed_track_ids.size(), 3u);
  EXPECT_EQ(result->confirmed_track_likelihoods.size(), 3u);
  EXPECT_EQ(result->track_ids[1][2], 22u);
  EXPECT_FLOAT_EQ(result->track_likelihoods[1][1], 1.5f);
  EXPECT_EQ(result->confirmed_track_ids[2][1], 31u);
  EXPECT_FLOAT_EQ(result->confirmed_track_likelihoods[2][2], 0.25f);
  EXPECT_EQ(result->faces.size(), 1u);
}

TEST(MeshSerialization, MeshBinary) {
  Mesh mesh(true, true, true, false, true, true);
  mesh.points.push_back(Eigen::Vector3f::Zero());
  mesh.points.push_back(Eigen::Vector3f::Zero());
  mesh.points.push_back(Eigen::Vector3f::Zero());
  mesh.colors.push_back({10, 20, 30, 255});
  mesh.labels.push_back(2);
  mesh.labels.push_back(8);
  mesh.stamps.push_back(0);
  mesh.stamps.push_back(10);
  mesh.stamps.push_back(20);
  mesh.faces.push_back({{1, 2, 3}});
  addTrackData(mesh);
  addConfirmedTrackData(mesh);

  std::vector<uint8_t> buffer;
  mesh.serializeToBinary(buffer);
  auto result = Mesh::deserializeFromBinary(buffer.data(), buffer.size());

  ASSERT_TRUE(result);
  EXPECT_EQ(result->points.size(), 3u);
  EXPECT_EQ(result->colors.size(), 1u);
  EXPECT_EQ(result->labels.size(), 2u);
  EXPECT_EQ(result->stamps.size(), 3u);
  EXPECT_TRUE(result->has_track_data);
  EXPECT_TRUE(result->has_confirmed_track_data);
  EXPECT_EQ(result->track_ids.size(), 3u);
  EXPECT_EQ(result->track_likelihoods.size(), 3u);
  EXPECT_EQ(result->confirmed_track_ids.size(), 3u);
  EXPECT_EQ(result->confirmed_track_likelihoods.size(), 3u);
  EXPECT_EQ(result->track_ids[0][0], 10u);
  EXPECT_FLOAT_EQ(result->track_likelihoods[2][0], 3.0f);
  EXPECT_EQ(result->confirmed_track_ids[1][1], 21u);
  EXPECT_FLOAT_EQ(result->confirmed_track_likelihoods[0][0], 0.75f);
  EXPECT_EQ(result->faces.size(), 1u);
}

}  // namespace spark_dsg
