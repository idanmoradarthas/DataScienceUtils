IF EXIST dist del dist
python -m build --sdist --wheel
twine upload dist/*