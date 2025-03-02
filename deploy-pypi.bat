IF EXIST dist rmdir /s /q dist
hatch build
hatch publish