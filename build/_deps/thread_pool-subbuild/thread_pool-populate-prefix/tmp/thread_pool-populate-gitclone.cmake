
if(NOT "/home/tianchen/racon/build/_deps/thread_pool-subbuild/thread_pool-populate-prefix/src/thread_pool-populate-stamp/thread_pool-populate-gitinfo.txt" IS_NEWER_THAN "/home/tianchen/racon/build/_deps/thread_pool-subbuild/thread_pool-populate-prefix/src/thread_pool-populate-stamp/thread_pool-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/tianchen/racon/build/_deps/thread_pool-subbuild/thread_pool-populate-prefix/src/thread_pool-populate-stamp/thread_pool-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/home/tianchen/racon/build/_deps/thread_pool-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/tianchen/racon/build/_deps/thread_pool-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "https://github.com/rvaser/thread_pool" "thread_pool-src"
    WORKING_DIRECTORY "/home/tianchen/racon/build/_deps"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/rvaser/thread_pool'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout 4.0.0 --
  WORKING_DIRECTORY "/home/tianchen/racon/build/_deps/thread_pool-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '4.0.0'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/tianchen/racon/build/_deps/thread_pool-src"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/tianchen/racon/build/_deps/thread_pool-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/tianchen/racon/build/_deps/thread_pool-subbuild/thread_pool-populate-prefix/src/thread_pool-populate-stamp/thread_pool-populate-gitinfo.txt"
    "/home/tianchen/racon/build/_deps/thread_pool-subbuild/thread_pool-populate-prefix/src/thread_pool-populate-stamp/thread_pool-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/tianchen/racon/build/_deps/thread_pool-subbuild/thread_pool-populate-prefix/src/thread_pool-populate-stamp/thread_pool-populate-gitclone-lastrun.txt'")
endif()

