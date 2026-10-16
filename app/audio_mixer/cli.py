import argparse
import json

from .api import db, ffmpeg


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    k = sub.add_parser("key-create")
    k.add_argument("--name", default="local")
    k.add_argument("--scope", action="append", default=["audio:admin"])
    c = sub.add_parser("capabilities")
    a = sub.add_parser("import-mixkit")
    a.add_argument("json_path")
    args = p.parse_args()
    if args.cmd == "key-create":
        kid, secret = db.create_key(args.name, "", args.scope)
        print(json.dumps({"id": kid, "secret": secret}, ensure_ascii=False))
    elif args.cmd == "capabilities":
        print(json.dumps(ffmpeg.capabilities(), ensure_ascii=False))
    elif args.cmd == "import-mixkit":
        with open(args.json_path, encoding="utf-8") as source:
            d = json.load(source)
        for c in d.get("categories", []):
            db.seed_category(c["id"], c.get("title", c["id"]))
        print(
            "Imported category metadata; audio binaries are not included in this dataset."
        )


if __name__ == "__main__":
    main()
