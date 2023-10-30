import unittest
import random

from hf_files.falcon_v00.handler import EndpointHandler as EndpointHandlerFalcon0

class TestHandler(unittest.TestCase):

    def test_handlers(self):
        for classe in [
            EndpointHandlerFalcon0
        ]:
            self.do_test_handler(classe)


    def do_test_handler(self, mother_class):

        class MockHandler(mother_class, unittest.TestCase):
            def __init__(self, path):
                pass

            def pipeline(self, inputs, **parameters):
                if isinstance(inputs, str):
                    return self.pipeline([inputs], **parameters)[0]

                self.assertTrue(isinstance(inputs, list))
                self.assertTrue([isinstance(text, str) for text in inputs])

                suffix = "\n[Intervenant 10:] Au revoir"
                return [[{"generated_text": text+suffix}] for text in inputs]

        self.maxDiff = None

        mock_handler = MockHandler(None)

        inputs_spks = [
            ("[Intervenant 1:] Bonjour\n[Intevenant 2:] Bonjour Jean, je\n", None),
            ("[spk1:] Bonjour\n[spk2:] Bonjour Jean, je", "spk"),
            ("[speaker001:] Bonjour\n[speaker002:] Bonjour Jean, je", "speaker00"),
        ]

        expected_outputs = []
        for text, spk in inputs_spks:
            output = text+"\n[Intervenant 10:] Au revoir"
            output = mock_handler({"inputs": text})
            expected_text = text+"\n[Intervenant 10:] Au revoir"
            if spk:
                expected_text = expected_text.replace("Intervenant ", spk)
            expected_output = [{"generated_text": expected_text}]
            self.assertEqual(output, expected_output)
            expected_outputs += expected_output

        inputs = [text for text, _ in inputs_spks]
        outputs = mock_handler({"inputs": inputs})
        self.assertEqual(outputs, expected_outputs)
