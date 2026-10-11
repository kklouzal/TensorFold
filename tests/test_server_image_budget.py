"""Frontend-specific default budgets survive count overrides; explicit budgets stay explicit."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.server.prompts import prepare_images
from tensorfold.vision import images


class ImageBudget(unittest.TestCase):
    def prepare(self, limits=images.DEFAULT_LIMITS, *, explicit=None):
        frontend = SimpleNamespace(image_limits=replace(images.DEFAULT_LIMITS, max_images=50,
                                   max_total_encoded_bytes=64 << 20, max_total_pixels=128 << 20))
        seen = {}

        def split(messages, **kwargs):
            seen['limits'] = kwargs['limits']
            return messages, []

        def prepare(rendered, loaded, **kwargs):
            seen['budget'] = kwargs
            return SimpleNamespace(token_ids=[1, 2, 3])

        frontend.prepare = prepare
        with patch.object(images, 'split_images', side_effect=split), \
             patch.object(images, 'load_images', return_value=[]):
            result = prepare_images(frontend, [], str, limits=limits, max_visual_tokens=explicit)
        self.assertEqual(result.tokens, [1, 2, 3])
        self.assertEqual(seen['limits'].max_total_encoded_bytes, 64 << 20)
        self.assertEqual(seen['limits'].max_total_pixels, 128 << 20)
        return seen

    def test_frontend_default_and_count_only_override_do_not_replace_token_default(self):
        seen = self.prepare()
        self.assertEqual(seen['limits'].max_images, 50)
        self.assertNotIn('max_visual_tokens', seen['budget'])
        seen = self.prepare(images.ImageLimits(max_images=8))
        self.assertEqual(seen['limits'].max_images, 8)
        self.assertNotIn('max_visual_tokens', seen['budget'])

    def test_nondefault_limit_reaches_frontend_and_preserves_source_byte_bounds(self):
        seen = self.prepare(images.ImageLimits(max_visual_tokens=16384))
        self.assertEqual(seen['budget']['max_visual_tokens'], 16384)

    def test_explicit_4096_is_forwarded_even_when_equal_to_shared_default(self):
        seen = self.prepare(explicit=4096)
        self.assertEqual(seen['budget']['max_visual_tokens'], 4096)
