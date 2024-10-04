"""
TODO write pod config based on profiled results
"""
import argparse
from pathlib import Path
import os
import csv
from jinja2 import Template, Environment, BaseLoader

import re

def regex_replace(s, pattern, replacement):
    return re.sub(pattern, replacement, s)



class Configurator():
    def __init__(self, args):
        self.args = args
     
    def writeJobYaml(self):
        #read input, template
        # Load the Jinja2 template
        
        template_path = os.path.join(os.getcwd(), self.args.template_file)
        with open(template_path, 'r') as template_f:
            jinja_template = Template(template_f.read())

        # Read the CSV file and parse its contents
        csv_data = []
        with open(self.args.input_file, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                row['JOB'] = regex_replace(row['JOB'], r'[^a-z0-9]+', '-')
                row['model_name'] = regex_replace(row['model_name'], r'[^a-z0-9]+', '-')
                row['RUN'] = self.args.run
                csv_data.append(row)

        #print(csv_data)
        # Render the Jinja2 template with the CSV data
        rendered_template = jinja_template.render(csv_data=csv_data)

        # Write the rendered template to a single output file
        # Create the output directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, args.output_name)
        with open(output_file, 'w') as output_f:
            output_f.write(rendered_template)

        print(f"Merged pod configurations saved to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str
    )
    parser.add_argument(
        "--template_file",
        type=Path,
        default="../template/template.jinja2"
    )

    parser.add_argument(
        "--run",
        type=int
    )

    #parser.add_argument(
    #    "--output_file",
    #    type=str,
    #)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="../k8sJobs"
    )

    parser.add_argument(
        "--output_name",
        type=str
    )

    args = parser.parse_args()
    configurator = Configurator(args)
    configurator.writeJobYaml()
    

    
    
    
