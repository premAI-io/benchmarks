#!/bin/bash

# This script downloads files from a list of URLs specified in a JSON file.
# The JSON file should contain an array of objects, each with a 'url', 'file', and 'folder' property.
# The 'url' property is the URL of the file to download, the 'file' property is the name of the file,
# and the 'folder' property is the directory where the file should be downloaded to.
# The script checks if the file already exists before downloading it.
#
# Usage: ./download.sh <json_file>
#
# Example:
# ./download.sh models.json
#
# This script requires the 'jq' tool to parse the JSON file.
# run brew install jq (MacOS)
# run apt-get install jq (Linux)

# Function to download a file
download_file() {
 local url=$1
 local file=$2
 local dir=$3

 # Create the directory if it does not exist
 mkdir -p $dir

 # Download the file
 wget -N $url -O $dir/$file
}

# Check if the JSON filename was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <json_file>"
  exit 1
fi

# Check if the JSON file exists
if [ ! -f "$1" ]; then
 echo "Error: JSON file does not exist."
 exit 1
fi

# Read the JSON file
json=$(cat $1)

# Parse the JSON file and iterate over its elements
echo $json | jq -r '.[] | @base64' | while read i; do
 _jq() {
     echo ${i} | base64 --decode | jq -r ${1}
 }

 url=$(_jq '.url')
 file=$(_jq '.file')
 folder=$(_jq '.folder')

 # Check if the file already exists
 if [ ! -f "$folder/$file" ]; then
  echo "Downloading: $url to $dir/$file"
  download_file $url $file $folder
 else
  echo "File already exists at $folder/$file"
 fi
done
