//
// unit test of morton_codes.h

#include <algorithm>
#include <bitset>
#include <vector>
#include <cstdint>

#include "OrthoTree/helpers/MortonHelper.h"
#include "gtest/gtest.h"

using namespace ippl;

TEST(MortonCodesTest, Encode3D)
{
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  grid_coordinate_template<Dim> coords = { 1, 2, 4 };
  morton_code code = morton.encode(coords, 3);
  morton_code expected = 0b1000100010011;
  EXPECT_EQ(code, expected);
}

TEST(MortonCodesTest, Encode2D) {
    static constexpr size_t Dim = 2;
    const size_t max_depth      = 1;
    Morton<Dim> morton(max_depth);

    grid_coordinate_template<Dim> coord1 = {0, 0};  // top-left corner
    grid_coordinate_template<Dim> coord2 = {1, 0};  // top-middle corner
    grid_coordinate_template<Dim> coord3 = {0, 1};  // middle-left corner
    grid_coordinate_template<Dim> coord4 = {1, 1};  // middle-middle corner

    morton_code code1 = morton.encode(coord1, max_depth);
    morton_code code2 = morton.encode(coord2, max_depth);
    morton_code code3 = morton.encode(coord3, max_depth);
    morton_code code4 = morton.encode(coord4, max_depth);

    const morton_code expected1 = 0b001;
    const morton_code expected2 = 0b011;
    const morton_code expected3 = 0b101;
    const morton_code expected4 = 0b111;

    EXPECT_EQ(code1, expected1);
    EXPECT_EQ(code2, expected2);
    EXPECT_EQ(code3, expected3);
    EXPECT_EQ(code4, expected4);
}

TEST(MortonCodesTest, Decode3D) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code code = 0b1000100010011;
  Morton<Dim>::grid_coordinate coords = morton.decode(code);
  Morton<Dim>::grid_coordinate expected = { 1, 2, 4 };

  bool is_same = true;
  for ( size_t i = 0; i < Dim; ++i ) {
    if ( coords[i] != expected[i] ) {
      is_same = false;
      std::cout << coords[i] << ", " << expected[i] << std::endl;
      break;
    }
  }

  EXPECT_EQ(is_same, true);
}

TEST(MortonCodesTest, isDescendantTest) {

  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  int64_t root = 0;
  int64_t parent = 0b111;
  int64_t child1 = 0b01;
  int64_t child2 = 0b0011000;
  int64_t child3 = 0b1111111000;

  EXPECT_FALSE(morton.is_descendant(child1, parent));
  EXPECT_TRUE(morton.is_descendant(child2,parent));
  EXPECT_FALSE(morton.is_descendant(child3, parent));

  EXPECT_TRUE(morton.is_descendant(child1, root));
  EXPECT_TRUE(morton.is_descendant(child2, root));
  EXPECT_TRUE(morton.is_descendant(child3, root));
  EXPECT_TRUE(morton.is_descendant(parent, root));
}

TEST(MortonCodesTest, GetDeepestFirstChildTest)
{
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code root = 0;
  morton_code child = morton.get_deepest_first_descendant(root);
  morton_code expected = 8;
  EXPECT_EQ(child, expected);
}


TEST(MortonCodesTest, GetDeepestLastChildTest) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code root = 0;
  morton_code child = morton.get_deepest_last_descendant(root);
  morton_code expected = morton.encode({ 255, 255, 255 }, 8);
  EXPECT_EQ(child, expected);
}

TEST(MortonCodesTest, GetNearestCommonAncestorTest) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code code_a = morton.encode({ 124, 124, 124 }, 6);
  morton_code code_b = morton.encode({ 0, 0, 0 }, 8);
  morton_code ancestor = morton.get_nearest_common_ancestor(code_a, code_b);
  morton_code expected = morton.encode({ 0, 0, 0 }, 1);
  EXPECT_EQ(ancestor, expected);
}

TEST(MortonCodesTest, GetChildrenTest) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code parent = morton.encode({ 126, 126, 126 }, 7);

  Kokkos::ScopeGuard kokkos_guard;

  Kokkos::View<morton_code*> children = morton.get_children(parent);
  std::vector<morton_code> expected;
  for ( grid_t i = 0; i < 2; i++ ) {
    for ( grid_t j = 0; j < 2; j++ ) {
      for ( grid_t k = 0; k < 2; k++ ) {
        expected.push_back(morton.encode({126 + i, 126 + j, 126 + k}, 8));
      }
    }
  }

  std::sort(expected.begin(), expected.end());
  
  EXPECT_EQ(children.size(), expected.size());
  for (unsigned i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(children(i), expected[i]);
  }
}

TEST(MortonCodesTest, GetParentTest) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code child = morton.encode({ 126, 126, 126 }, 7);
  morton_code parent = morton.get_parent(child);
  morton_code expected = morton.encode({ 124, 124, 124 }, 6);
  EXPECT_EQ(parent, expected);
}

