if [ "$#" != "2" ]; then
	echo "Usage: \"./$(basename \"$0\") MODEL_FOLDER OUTPUT_FOLDER"
	exit 1
fi

python3 -m transformers.onnx \
	--model="$1" \
	--feature="question-answering" \
	"$2"
