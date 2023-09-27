import unittest
import random

from utils.text import augmented_texts_generator

class TestFormatText(unittest.TestCase):

    def test_augmentation(self):

        for itest, (text, normalized_text, maximum) in enumerate([
            (
                "[speaker001:] Tu me fais rire [LAUGHTER]. Je chante [SINGING]? [Claire Michel:] Il y a un bruit [NOISE], je l'ai dit à [PII].",
                "[speaker001:] Tu me fais rire [laughter]. Je chante ? [claire michel:] Il y a un bruit [noise], je l'ai dit à [pii].",
                4,
            ),
            (
                "[speaker001:] Tu me fais rire. Je chante ? [Claire Michel:] Il y a un bruit, je l'ai dit à Ted.",
                "[speaker001:] Tu me fais rire. Je chante ? [claire michel:] Il y a un bruit, je l'ai dit à Ted.",
                3,
            ),
            (
                "[speaker001:] Tu me fais rire [LAUGHTER]. Je chante [SINGING]? [speaker002:] Il y a un bruit [NOISE], je l'ai dit à [PII].",
                "[speaker001:] Tu me fais rire [laughter]. Je chante ? [speaker002:] Il y a un bruit [noise], je l'ai dit à [pii].",
                3,
            ),
            (
                "[speaker001:] Tu me fais rire Je chante [SINGING] [speaker002:] Il y a un bruit je l'ai dit à Ted",
                "[speaker001:] Tu me fais rire Je chante [speaker002:] Il y a un bruit je l'ai dit à Ted",
                1,
            ),
            (
                "[speaker001:] tu me fais rire. je chante [SINGING] ? [speaker002:] il y a un bruit, je l'ai dit à ted",
                "[speaker001:] tu me fais rire. je chante ? [speaker002:] il y a un bruit, je l'ai dit à ted",
                1,
            ),
            (
                "[speaker001:] tu me fais rire je chante [SINGING] [speaker002:] il y a un bruit je l'ai dit à ted",
                "[speaker001:] tu me fais rire je chante [speaker002:] il y a un bruit je l'ai dit à ted",
                0,
            ),
        ]):

            for level in 0, 1, 2, 3, 4, 5:

                # Note: Ted is the one generated by names with the seed used below
                extreme_regex = r"\[speaker001:\] tu me fais rire je chante \[speaker002:\] il y a un bruit je l'ai dit à ted"
                random.seed(51)

                augmented_texts = list(augmented_texts_generator(text, level))
                msg_augmented_texts= '\n  * '.join(augmented_texts)
                msg = f"\n{itest=}\n{level=}\n{text=}\naugmented_texts:\n  * {msg_augmented_texts}"
                self.assertEqual(len(augmented_texts), min(maximum+1, level+1), msg=msg)    # Expected number of generated text
                self.assertEqual(len(augmented_texts), len(set(augmented_texts)), msg=msg)  # All generated texts are different
                self.assertEqual(augmented_texts[0], normalized_text, msg=msg)              # First text is the normalized text
                if len(augmented_texts) > 1:
                    self.assertRegex(augmented_texts[-1], extreme_regex, msg=msg)                 # The deepest normalization is always included among the augmented texts

