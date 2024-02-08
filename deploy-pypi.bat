IF EXIST dist del dist
python -m build --sdist --wheel
twine upload --config-file .pypirc -r data-science-utils dist\*