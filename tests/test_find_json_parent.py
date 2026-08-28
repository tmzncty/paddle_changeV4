import os
import tempfile
import unittest

from find_json_parent import find_unique_json_parent_paths


class FindJsonParentTests(unittest.TestCase):
    def test_returns_unique_sorted_directories_containing_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            book_a = os.path.join(temp_dir, "json_output", "A")
            book_b = os.path.join(temp_dir, "json_output", "B")
            ignored = os.path.join(temp_dir, "temp_images", "C")
            os.makedirs(book_a)
            os.makedirs(book_b)
            os.makedirs(ignored)

            with open(os.path.join(book_a, "page_00001.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(book_a, "page_00002.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(book_b, "page_0001_result.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with open(os.path.join(ignored, "page_00001.png"), "w", encoding="utf-8") as handle:
                handle.write("not really an image")

            result = find_unique_json_parent_paths(temp_dir)

            self.assertEqual(result, sorted([book_a, book_b]))

    def test_missing_root_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "does-not-exist")
            self.assertEqual(find_unique_json_parent_paths(missing), [])

    def test_non_json_files_do_not_create_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "book"))
            with open(os.path.join(temp_dir, "book", "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("hello")

            self.assertEqual(find_unique_json_parent_paths(temp_dir), [])


if __name__ == "__main__":
    unittest.main()
