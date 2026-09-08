"""Skirmisher coherency and casualty boundaries, Rulebook p. 184 and FAQ."""

import unittest

from skirmish import (casualty_indices, coherent_groups, coherency_error,
                      layout_positions, separated_positions, swept_base_overlaps, targeted_replacement)


def boxes_at(*positions):
    return [(position, 0, 0.5, 0.5, 0) for position in positions]


class CoherencyTests(unittest.TestCase):
    def test_inclusive_one_inch_base_edge_gap(self):
        self.assertIsNone(coherency_error(boxes_at(0, 2)))
        self.assertIsNotNone(coherency_error(boxes_at(0, 2.001)))

    def test_two_disconnected_pairs_are_not_a_unit(self):
        boxes = boxes_at(0, 1.5, 6, 7.5)
        self.assertEqual(coherent_groups(boxes), [{0, 1}, {2, 3}])
        self.assertIsNotNone(coherency_error(boxes))

    def test_touching_and_overlapping_bases_are_not_loose(self):
        self.assertIsNotNone(coherency_error(boxes_at(0, 1)))
        self.assertIsNotNone(coherency_error(boxes_at(0, 0.5)))
        self.assertIsNone(coherency_error(boxes_at(0, 1), loose=False))

    def test_lone_survivor_and_empty_unit(self):
        self.assertIsNone(coherency_error(boxes_at(0)))
        self.assertIsNone(coherency_error([]))

    def test_rotated_bases_use_edges_not_centres(self):
        boxes = [(0, 0, 1, 0.5, 90), (0, 3, 1, 0.5, 90)]
        self.assertIsNone(coherency_error(boxes))

    def test_layouts_are_connected_and_repeatable(self):
        for count in range(1, 26):
            for columns in (1, 3, 7):
                with self.subTest(count=count, columns=columns):
                    positions = layout_positions(count, 0.98, 1.96, columns)
                    boxes = [(point[0], point[1], 0.49, 0.98, 0) for point in positions]
                    self.assertIsNone(coherency_error(boxes))
                    self.assertEqual(positions, layout_positions(count, 0.98, 1.96, columns))

    def test_invalid_spacing_rejected(self):
        for gap in (0, -1, 1.01):
            with self.assertRaises(ValueError):
                layout_positions(4, 1, 1, gap=gap)


class SeparationTests(unittest.TestCase):
    def test_compact_rank_separates_without_reshuffling(self):
        for count in (1, 5, 20, 50):
            with self.subTest(count=count):
                boxes = [(index * 0.98, 0, 0.49, 0.98, 0) for index in range(count)]
                positions = separated_positions(boxes)
                after = [(*position, *box[2:]) for position, box in zip(positions, boxes)]
                self.assertIsNone(coherency_error(after))
                self.assertEqual(positions, sorted(positions))
                for before, position in zip(boxes, positions):
                    self.assertAlmostEqual(position[0], before[0], delta=0.003)
                    self.assertEqual(position[1], before[1])

    def test_already_loose_bases_and_single_survivor_do_not_move(self):
        for boxes in ([], boxes_at(3.456), boxes_at(0, 1.6, 3.2)):
            self.assertEqual(separated_positions(boxes), [box[:2] for box in boxes])

    def test_larger_character_separates_with_touching_rank(self):
        boxes = [(0, 0, 1, 1, 0), (1.5, 0.5, 0.5, 0.5, 0), (1.5, -0.5, 0.5, 0.5, 0)]
        positions = separated_positions(boxes)
        self.assertIsNone(coherency_error([(*position, *box[2:]) for position, box in zip(positions, boxes)]))
        for box, position in zip(boxes, positions):
            self.assertAlmostEqual(box[0], position[0], delta=0.001)
            self.assertAlmostEqual(box[1], position[1], delta=0.001)

    def test_overlapping_bases_are_not_silently_reformed(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            separated_positions(boxes_at(0, 0.5))


class CasualtyTests(unittest.TestCase):
    def test_last_node_bridge_is_not_removed(self):
        boxes = boxes_at(0, 3, 1.5)
        self.assertEqual(casualty_indices(boxes, 1), [1])

    def test_batch_keeps_one_group_without_repositioning(self):
        boxes = boxes_at(0, 3, 1.5, 4.5)
        removed = casualty_indices(boxes, 3)
        self.assertEqual(len(removed), 3)
        remaining = [box for index, box in enumerate(boxes) if index not in removed]
        self.assertIsNone(coherency_error(remaining))

    def test_joined_character_is_not_an_ordinary_casualty(self):
        self.assertEqual(casualty_indices(boxes_at(0, 3, 1.5), 2, protected=[2]), [1, 0])

    def test_targeted_bridge_is_replaced_from_an_endpoint(self):
        boxes = boxes_at(0, 3, 1.5)
        self.assertEqual(targeted_replacement(boxes, 2, [0, 1]), 1)

    def test_targeted_endpoint_needs_no_replacement(self):
        self.assertIsNone(targeted_replacement(boxes_at(0, 1.5, 3), 2, [0, 1]))


class SweptBaseTests(unittest.TestCase):
    def test_edge_crossing_is_detected_without_centre_crossing(self):
        start, end = boxes_at(0, 5)
        self.assertTrue(swept_base_overlaps(start, end, (3, 0.6, 0.1, 0.2, 0)))
        self.assertFalse(swept_base_overlaps(start, end, (3, 1, 0.1, 0.2, 0)))

    def test_diagonal_sweep_does_not_fill_bounding_rectangle(self):
        start = (0, 0, 0.5, 0.5, 0)
        end = (5, 5, 0.5, 0.5, 0)
        self.assertFalse(swept_base_overlaps(start, end, (0, 5, 0.5, 0.5, 0)))
        self.assertTrue(swept_base_overlaps(start, end, (2.5, 2.5, 0.5, 0.5, 0)))


if __name__ == '__main__':
    unittest.main()