"""Initialize the database: python -m roboego_hand_vis.database init."""

import argparse

from .database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL data tables")
    parser.add_argument("command", choices=["init"])
    args = parser.parse_args()
    from .connection_config import database_url_or_error
    url = database_url_or_error(parser)
    database = Database.from_url(url)
    try:
        if args.command == "init":
            database.initialize()
            print("Initialized l1data, l2data, l3data")
    finally:
        database.close()


if __name__ == "__main__":
    main()
