#!/bin/bash

API_KEY="1ad6881be2d3c1f2bba008aec8b043ba"
TARGET_DIR="test"

mkdir -p "$TARGET_DIR"

for idx in 0 104 144 600; do
    OUTFILE="${TARGET_DIR}/snapshot-99.${idx}.hdf5"
    echo "⬇️ Downloading snapshot-99.${idx}.hdf5..."

    curl -L --fail --insecure -H "api-key: ${API_KEY}" \
        "https://www.tng-project.org/api/TNG300-1/files/snapshot-99.${idx}.hdf5/download" \
        -o "${OUTFILE}"

    if [ $? -ne 0 ]; then
        echo "❌ Failed to download snapshot-99.${idx}.hdf5"
        rm -f "${OUTFILE}"  # 잘못된 파일 삭제
    else
        echo "✅ Successfully downloaded snapshot-99.${idx}.hdf5"
    fi
done

echo "✅ All done."
