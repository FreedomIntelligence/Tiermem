"""Run with: python -m unittest -v check_contacts"""

import unittest
from pathlib import Path

from contacts import load_contacts


FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = [{"customer_id": "00073", "name": "Chen, Mei", "email": "mei@example.com"}]


class ContactImportTests(unittest.TestCase):
    def test_plain_utf8(self):
        self.assertEqual(load_contacts(FIXTURES / "plain.csv"), EXPECTED)

    def test_excel_utf8_bom(self):
        self.assertEqual(load_contacts(FIXTURES / "excel_export.csv"), EXPECTED)

    def test_preserves_leading_zero_and_quoted_comma(self):
        row = load_contacts(FIXTURES / "plain.csv")[0]
        self.assertEqual(row["customer_id"], "00073")
        self.assertEqual(row["name"], "Chen, Mei")
        print(f"Regression values: customer_id={row['customer_id']!r}, name={row['name']!r}", flush=True)


if __name__ == "__main__":
    unittest.main()
