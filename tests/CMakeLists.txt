function(add_sampler_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE isomorphism)
endfunction()

# Register test executables here:
add_sampler_test(test_basic)
