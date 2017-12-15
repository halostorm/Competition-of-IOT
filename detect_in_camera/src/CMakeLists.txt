add_library( my_detect SHARED
    config.cpp
)

target_link_libraries( my_detect
    ${THIRD_PARTY_LIBS}
)
