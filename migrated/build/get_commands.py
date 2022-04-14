import json
from pathlib import Path
import shlex
with open("compile_commands.json", "r") as f:
    s = json.load(f)
commands = []
cuda_root = Path("/home/tianchen/racon")
migrated_root = Path("/home/tianchen/racon/migrated")
cuda_install_root = Path("/usr/local/cuda-11.4")
output_change_dict = {}
for entry in s:
    c = entry["command"]
    params = shlex.split(c)
    
    if "file" in entry:
        # change the source file name and dir
        source_file = Path(params[-1])
        file_name = source_file.parts[-1]
        case = 0
        if file_name.endswith(".cu"):
            new_file_name = file_name[:-2] + "dp.cpp"
            case = 1
        elif file_name.endswith(".cpp") or file_name.endswith(".cc") or file_name.endswith(".cxx") or file_name.endswith(".C"):
            if migrated_root.joinpath(source_file.relative_to(cuda_root)).exists():
                new_file_name = file_name
                case = 2
            else:
                new_file_name = file_name + ".dp.cpp"
                case = 3
        new_file_path = migrated_root.joinpath(source_file.relative_to(cuda_root)).resolve().parent.joinpath(new_file_name)
        params[-1] = str(new_file_path)

        # change output file name and dir
        output_flag = params.index("-o")
        output_file = Path(params[output_flag + 1])
        if not output_file.is_absolute():
            output_file = Path(entry["directory"]).joinpath(output_file).resolve()
        else:
            output_file = output_file.resolve()

        new_output_file = migrated_root.joinpath(output_file.relative_to(cuda_root))
        new_output_dir = new_output_file.parent
        new_output_file_name = new_output_file.parts[-1]
        if case == 1:
            new_output_file_name = new_output_file_name[:-5] + ".dp.cpp.o"
        elif case == 3:
            new_output_file_name = new_output_file_name[:-2] + ".dp.cpp.o"
        # create output directory
        if not new_output_dir.exists():
            new_output_dir.mkdir(parents=True)
        new_output_file = new_output_dir.joinpath(new_output_file_name)
        output_change_dict[output_file] = new_output_file
        params[output_flag + 1] = str(new_output_file)

        # change compiler
        # if case == 1 or case == 3:
        #     params[0] = "dpcpp"
        # else:
        #     params[0] = "icpx"
        params[0] = "dpcpp"
        
        # about -I
        new_params = []
        for param in params:
            if param.startswith("-I"):
                include_path = Path(param[2:]).resolve()
                # change the dir
                if cuda_root in include_path.parents:
                    new_include_path = migrated_root.joinpath(include_path.relative_to(cuda_root))
                else:
                    new_include_path = include_path
                # remove cuda dependency
                if cuda_install_root in new_include_path.parents:
                    continue
                new_param = "-I" + str(new_include_path)
            elif param.startswith("-std=c++"):
                new_param = "-std=c++20"
            else:
                new_param = param
            new_params.append(new_param)

        # add stdc++9 macros for dpl problem
        flag = new_params.index("-o")
        macros = "-DPSTL_USE_PARALLEL_POLICIES=0"
        new_params.insert(flag, macros)
        new_params.insert(flag, "-Wno-tautological-constant-compare")

        new_params = [param for param in new_params if param not in ("-pedantic", "-Wall", "-Wextra")]
        
        commands.append(shlex.join(new_params))
    
    elif c.startswith("ar qc"):
        new_params = ["ar", "qc"]
        directory = Path(entry["directory"]).resolve()
        for param in params[2:]:
            old_path = Path(param)
            if not old_path.is_absolute():
                old_path = directory.joinpath(old_path).resolve()
                if str(old_path).endswith(".a"):
                    new_path = migrated_root.joinpath(old_path.relative_to(cuda_root))
                    if not new_path.parent.exists():
                        new_path.parent.mkdir(parents=True)
                    output_change_dict[old_path] = new_path
                else:
                    new_path = output_change_dict[old_path]
            elif cuda_root in old_path.parents:
                new_path = migrated_root.joinpath(old_path.relative_to(cuda_root))
            else:
                new_path = old_path
                
            new_params.append(str(new_path))
        commands.append(shlex.join(new_params))
    
    elif c.startswith("ld"):
        new_params = ["dpcpp", "-o"]
        flag = params.index("-o") + 1
        directory = Path(entry["directory"]).resolve()
        for param in params[flag:]:
            old_path = Path(param)
            if not old_path.is_absolute():
                old_path = directory.joinpath(old_path).resolve()
                if str(old_path).endswith(".a") or str(old_path).endswith(".o"):
                    new_path = output_change_dict[old_path]
                else:
                    new_path = migrated_root.joinpath(old_path.relative_to(cuda_root))
            elif cuda_install_root in old_path.parents:
                continue
            elif cuda_root in old_path.parents:
                new_path = migrated_root.joinpath(old_path.relative_to(cuda_root))
            else:
                new_path = old_path
            if str(new_path).startswith("/usr/lib/gcc"):
                continue
            new_params.append(str(new_path))
        new_params.append("-lpthread")
        commands.append(shlex.join(new_params))



    else:
        commands.append(c)
    
    
content = "\n".join(commands)
with open("build.sh", "w") as f:
    f.write(content)

