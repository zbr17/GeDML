echo "Clear all __pycache__"
find . -name '__pycache__' -type d -exec rm -rf {} \;

echo "Copy GeDML to a new folder"
cp -r $WORKSPACE/code/GeDML $WORKSPACE/code/upload
cd ..
ls
cd $WORKSPACE/code/upload
ls

echo "Build"
python -m build
ls

echo "Upload"
twine upload dist/*
ls
read -p "pause"

echo "Delete the tmp folder"
sudo rm -r $WORKSPACE/code/upload

echo "Finish!"