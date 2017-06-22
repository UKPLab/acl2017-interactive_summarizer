import argparse
import json

from os import path
import xml.etree.ElementTree as ET

from utils.reader import read_file
from utils.writer import create_dir, write_to_file


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    # DUC2001, DUC2002, DUC2004
    parser.add_argument('-i', '--input', help="Input file", type=str, required=True)

    # ../data
    parser.add_argument('-o', '--output', help="Target directory", type=str, required=False,
                        default=path.join("~", ".ukpsummarizer", "output", "topics"))

    # iobasedir
    # parser.add_argument('-i', '--iobase',
    #                     help="base directory",
    #                     type=str,
    #                     required=False,
    #                     default=path.normpath(path.join(path.expanduser("~"), ".ukpsummarizer")))
    args = parser.parse_args()

    tde = TaskExtractor(args.output)
    tde.process(args.input)


class TaskExtractor:
    def __init__(self, output_directory):
        create_dir(output_directory)
        self.output_directory = create_dir(output_directory)


    def process(self, input_file, type="DUC2006"):
        file_data = read_file(input_file)
        try:
            root = ET.fromstring(file_data)
        except ET.ParseError:
            # some DUC topic xmls are broken, try to fix them:
            root = ET.fromstring("<xml>%s</xml>" % file_data)

        for child in root.findall("topic"):
            topic_description = self.parseTopic(child)
            self.create_processed(topic_description)

        # print topics

    def parseTopic(self, xml_element):
        doc_id = xml_element.find("num").text.strip()
        title = xml_element.find("title").text.strip()
        narrative = xml_element.find("narr").text.strip()

        return {"id": doc_id, "title": title, "narrative": narrative}

    def create_processed(self, topic_description):
        filename = path.join(create_dir(path.join(self.output_directory, topic_description["id"])), "task.json")
        retval = json.dumps(topic_description)
        write_to_file(retval, filename)




if __name__ == '__main__':
    main()
