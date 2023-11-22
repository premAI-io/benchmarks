#!/bin/bash
set -euo pipefail

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "  -p, --prompt        Prompt for benchmarks (default: 'Explain what is a transformer')"
    echo "  -r, --repetitions   Number of repetitions for benchmarks (default: 10)"
    echo "  -m, --max_tokens    Maximum number of tokens for benchmarks (default: 100)"
    echo "  -d, --device        Device for benchmarks (possible values: 'metal', 'gpu', and 'cpu', default: 'cpu')"
    echo "  -lf, --log_file     Logging file name."
    echo "  -md, --models_dir   Models directory."
    echo "  -h, --help          Show this help message"
    exit 1
}


download_models() {
    echo -e "\nDownloading models..."
    bash ./download.sh --models models.json --cache cache.log
}

check_jq() {
    if ! command -v jq &> /dev/null
    then
        echo -e "\njq is not installed."
        exit 1
    fi
}

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -r|--repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        -m|--max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            case "$DEVICE" in
                "cuda" | "metal" | "cpu")
                    ;;
                *)
                    echo "Invalid value for --device. Please use 'cuda', 'gpu' or 'cpu'."
                    print_usage
                    ;;
            esac
            shift 2
            ;;
        -lf|--log_file)
            LOG_FILENAME="$2"
            shift 2
            ;;
        -md|--models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

check_jq
download_models


PROMPT="${PROMPT:-"Explain what is a transformer"}"
REPETITIONS="${REPETITIONS:-10}"
MAX_TOKENS="${MAX_TOKENS:-100}"
DEVICE="${DEVICE:-'cpu'}"
LOG_FILENAME="${LOG_FILENAME:-"benchmark_$(date +'%Y%m%d%H%M%S').log"}"
MODELS_DIR="${MODELS_DIR:-"./models"}"

folders=$(find . -type d -name "bench_*")

for folder in $folders; do
    if [ -d "$folder" ]; then
        echo "Running benchmark $folder/bench.sh..."

        if ! bash "$folder/bench.sh" \
            --prompt "$PROMPT" \
            --repetitions "$REPETITIONS" \
            --max_tokens "$MAX_TOKENS" \
            --models_dir "$MODELS_DIR" \
            --log_file "$LOG_FILENAME" \
            --device "$DEVICE"; then
            echo "Error: Something went wrong in $folder/bench.sh"
        else
            echo "Success: $folder/bench.sh completed successfully"
        fi
    fi
done
