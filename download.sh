#!/bin/bash

################################################################################
# Script: download.sh
# Description: Downloads files from a list of URLs specified in a JSON file. 
# The JSON file should contain an array of objects, each with a 'url', 'file', 
# and 'folder' property. The script checks if the file already exists before 
# downloading it.
#
# Usage: ./download.sh --models <json_file> --cache <cache_file> --force-download
#
# Example:
# ./download.sh --models models.json --cache cache.log --force-download
################################################################################

set -euo pipefail

# Default values
models_file="$(pwd)/models.json"
cache_file="$(pwd)/cache.log"
force_download=false

# Function to download a file
download_file() {
 local url=$1
 local file=$2
 local dir=$3

 # Create the directory if it does not exist
 mkdir -p "$dir"

 # Download the file
 wget -N "$url" -O "$dir/$file"
 echo "$url" >> "$cache_file"
}

# Function to unzip a file
unzip_file() {
 local file=$1
 local dir=$2

 # Unzip the file
 unzip -o "$file" -d "$dir"

 # Move the unzipped files to the parent directory
 find "$dir" -mindepth 2 -type f -exec mv {} "$dir" \;
}

# Function to remove a file
remove_file() {
 local file=$1

 # Remove the file
 rm "$file"
}

# Argument parsing
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --models)
      models_file="$2"
      shift # past argument
      shift # past value
      ;;
    --cache)
      cache_file="$2"
      shift # past argument
      shift # past value
      ;;
    --force-download)
      force_download=true
      shift # past argument
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if the required arguments are provided
if [ -z "$models_file" ] || [ -z "$cache_file" ]; then
  echo "Usage: $0 --models <json_file> --cache <cache_file> [--force-download]"
  exit 1
fi

# Check if the JSON file exists
if [ ! -f "$models_file" ]; then
  echo "Error: JSON file '$models_file' does not exist."
  exit 1
fi

# Check if force download is enabled
if $force_download; then
  echo "Force download enabled. Removing all files in the models folder and cache file."
  rm -rf ./models/*
  > "$cache_file"
fi

# Read the JSON file
json=$(cat "$models_file")

# Parse the JSON file and iterate over its elements
echo "$json" | jq -r '.[] | @base64' | while read -r i; do
 _jq() {
     echo "${i}" | base64 --decode | jq -r "${1}"
 }

 url=$(_jq '.url')
 file=$(_jq '.file')
 folder=$(_jq '.folder')

 # Check if the URL is in the log file
 if ! grep -q "$url" "$cache_file"; then
   if [[ $file == *.zip ]]; then
     echo "Downloading and unzipping: $url to $folder"
     download_file "$url" "$file" "$folder"
     unzip_file "$folder/$file" "$folder"
     echo "Removing: $folder/$file"
     remove_file "$folder/$file"
   else
     echo "Downloading: $url to $folder/$file"
     download_file "$url" "$file" "$folder"
   fi
 fi
done
