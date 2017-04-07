import argparse
import sys


class HelpOnFailArgumentParser(argparse.ArgumentParser):
    """
    Prints help whenever the command-line arguments could not be parsed.
    """

    def error(self, message):
        sys.stderr.write('Error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)
