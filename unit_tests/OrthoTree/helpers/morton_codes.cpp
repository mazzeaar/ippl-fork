//
// unit test of morton_codes.h

#include <algorithm>
#include <bitset>
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

TEST(MortonCodesTest, GetNthChildTestFirstChild) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code parent = morton.encode({ 124, 124, 124 }, 6);
  morton_code child = morton.get_nth_child(parent, 0);
  morton_code expected = morton.encode({ 124, 124, 124 }, 7);
  EXPECT_EQ(child, expected);
}

TEST(MortonCodesTest, GetNthChildTestAll) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 3;
  Morton<Dim> morton(max_depth);

  morton_code parent = morton.encode({ 0, 0, 0 }, 0);
  morton_code child0 = morton.get_nth_descendant(parent, 3, 0);
  morton_code expected0 = morton.encode({ 0, 0, 0 }, 3);
  EXPECT_EQ(child0, expected0);
  morton_code child1 = morton.get_nth_descendant(parent, 3, 1);
  morton_code expected1 = morton.encode({ 7, 0, 0 }, 3);
  EXPECT_EQ(child1, expected1);


  morton_code child2 = morton.get_nth_descendant(parent, 3, 2);
  morton_code expected2 = morton.encode({ 0, 7, 0 }, 3);
  EXPECT_EQ(child2, expected2);

  morton_code child5 = morton.get_nth_descendant(parent, 3, 5);
  morton_code expected5 = morton.encode({ 7, 0, 7 }, 3);
  EXPECT_EQ(child5, expected5);
}

TEST(MortonCodesTest, GetNthDescendant2D) {
  static constexpr size_t Dim = 2;
  const size_t max_depth = 5;
  Morton<Dim> morton(max_depth);

  morton_code parent = morton.encode({ 0, 0 }, 0);
  morton_code child0 = morton.get_nth_descendant(parent, 3, 0);
  morton_code expected0 = morton.encode({ 0, 0 }, 3);
  EXPECT_EQ(child0, expected0);
  morton_code child1 = morton.get_nth_descendant(parent, 5, 1);
  morton_code expected1 = morton.encode({ 31, 0 }, 5);
  EXPECT_EQ(child1, expected1);
  morton_code child2 = morton.get_nth_descendant(parent, 3, 2);
  morton_code expected2 = morton.encode({ 0, 28 }, 3);
  EXPECT_EQ(child2, expected2);
  morton_code child3 = morton.get_nth_descendant(parent, 1, 3);
  morton_code expected3 = morton.encode({ 16, 16 }, 1);
  EXPECT_EQ(child3, expected3);
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
  vector_t<morton_code> children = morton.get_children(parent);
  vector_t<morton_code> expected;
  for ( grid_t i = 0; i < 2; i++ ) {
    for ( grid_t j = 0; j < 2; j++ ) {
      for ( grid_t k = 0; k < 2; k++ ) {
        expected.push_back(morton.encode({126 + i, 126 + j, 126 + k}, 8));
      }
    }
  }

  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(children, expected);
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

TEST(MortonCodesTest, GetChildIndexTest) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 8;
  Morton<Dim> morton(max_depth);

  morton_code parent = morton.encode({ 124, 124, 124 }, 6);
  morton_code child = morton.encode({ 124, 124, 124 }, 7);
  int index = morton.get_child_index(parent, child);
  int expected = 0;
  EXPECT_EQ(index, expected);

  morton_code child2 = morton.get_last_child(parent);
  int index2 = morton.get_child_index(parent, child2);
  int expected2 = 7;
  EXPECT_EQ(index2, expected2);

  morton_code child3 = morton.encode({ 0, 0, 0 }, 1);
  int index3 = morton.get_child_index(parent, child3);
  int expected3 = -1;
  EXPECT_EQ(index3, expected3);

  morton_code child4 = morton.encode({ 124, 124, 124 }, 8);
  int index4 = morton.get_child_index(parent, child4);
  int expected4 = -1;
  EXPECT_EQ(index4, expected4);
}

TEST(MortonCodesTest, GetSearchKeysTest) {
  static constexpr size_t Dim = 3;
  const size_t max_depth = 5;
  Morton<Dim> morton(max_depth);

  morton_code code = morton.encode({ 8, 8, 8 }, 2);
  vector_t<morton_code> keys = morton.get_search_keys(code);
  vector_t<morton_code> expected(7);

  expected[0] = morton.encode({ 16, 16, 16 }, 5);
  expected[1] = morton.encode({ 16, 16, 15 }, 5);
  expected[2] = morton.encode({ 16, 15, 16 }, 5);
  expected[3] = morton.encode({ 16, 15, 15 }, 5);
  expected[4] = morton.encode({ 15, 16, 16 }, 5);
  expected[5] = morton.encode({ 15, 16, 15 }, 5);
  expected[6] = morton.encode({ 15, 15, 16 }, 5);


  std::sort(expected.begin(), expected.end());
  std::sort(keys.begin(), keys.end());

  EXPECT_EQ(keys, expected);
}

