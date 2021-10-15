import csv
import argparse
import os

class ParserWithConvert:
    """
    We can employ this class to parser and convert args, which should work with ConfigHandler.

    Given the path of a .csv file which records the descriptions of args, 
    this class will output a `Parser` instance and convert dictionary.

    Args:
        csv_path (str):
            Path to the .csv file.
        name (str, optional):
            Name of the project.
    """
    def __init__(
        self,
        csv_path,
        name="",
    ):
        self.csv_path = csv_path
        self.name = name
        self.prefix = "config_"

        self.setting = None
    
    def render(self):
        """
        Generate parsed args and convert dictionary.

        Return:
            tuple. (args, convert)
        """
        self.read_csv()
        opt = self.get_parser()
        convert = self.get_convert()
        return opt, convert
    
    def read_csv(self):
        self.info_dict = {}
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            csv_f = csv.reader(f)
            headers = next(csv_f)
            headers = [item.strip() for item in headers]
            headers = [
                item if idx < 4 else self.prefix + item 
                for idx, item in enumerate(headers)
            ]
            for idx, row in enumerate(csv_f):
                row = [
                    eval(item) if item else ""
                    for item in row
                ]
                self.info_dict[idx] = dict(zip(headers, row))

    def get_parser(self):
        parser = argparse.ArgumentParser(self.name)
        for values in self.info_dict.values():
            if values["name"] == "--setting":
                raise KeyError("<--setting> is duplicated in the command line arguments!")
            
            # for action args
            if isinstance(values["type"], str) and "store" in values["type"]:
                if "required" == values["default"]:
                    parser.add_argument(values["name"], action=values["type"], required=True, help=values["help"])
                else:
                    parser.add_argument(values["name"], action=values["type"], default=values["default"], help=values["help"])
            # for list args
            elif isinstance(values["default"], list):
                if "required" == values["default"]:
                    parser.add_argument(values["name"], action=values["type"], required=True, help=values["help"], nargs="+")
                else:
                    parser.add_argument(values["name"], type=values["type"], default=values["default"], help=values["help"], nargs="+")
            # for other args
            else:
                if "required" == values["default"]:
                    parser.add_argument(values["name"], type=values["type"], required=True, help=values["help"])
                else:
                    parser.add_argument(values["name"], type=values["type"], default=values["default"], help=values["help"])
        # add <setting> argument
        parser.add_argument("--setting", type=str, help="Please input the setting")
        # get the args
        opt = parser.parse_args()
        self.setting = opt.setting
        return opt

    def get_convert(self):
        convert = {}
        for idx, values in self.info_dict.items():
            if values[self.prefix + self.setting]:
                convert[values["name"].strip('-')] = values[self.prefix + self.setting]
        return convert