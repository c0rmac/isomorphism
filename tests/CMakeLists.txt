function(add_sampler_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE isomorphism)
endfunction()

# Register test executables here:
add_sampler_test(test_basic)
add_sampler_test(test_qr)
add_sampler_test(test_qr1)
