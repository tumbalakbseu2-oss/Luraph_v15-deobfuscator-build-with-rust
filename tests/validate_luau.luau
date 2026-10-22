-- Lune Luau Syntax Validator
-- Validates that a file contains syntactically correct Luau code using loadstring.

local fs = require("@lune/fs")
local process = require("@lune/process")

if #process.args < 1 then
    print("USAGE: lune run validate_luau.luau <path_to_file.lua>")
    process.exit(1)
end

local target_path = process.args[1]

if not fs.isFile(target_path) then
    print("ERROR: File not found: " .. target_path)
    process.exit(1)
end

local content = fs.readFile(target_path)
local fn, err = loadstring(content)

if not fn then
    print("SYNTAX_ERROR: " .. tostring(err))
    process.exit(1)
else
    print("SYNTAX_VALID: " .. tostring(#content) .. " bytes")
    process.exit(0)
end
