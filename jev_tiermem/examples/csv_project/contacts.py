"""Small CSV importer used by the reproducible memory demo."""

import csv


def load_contacts(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return [
            {"customer_id": row["customer_id"], "name": row["name"], "email": row["email"]}
            for row in csv.DictReader(stream)
        ]
