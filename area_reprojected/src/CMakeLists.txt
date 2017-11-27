add_library( my_refind SHARED
    config.cpp
)

target_link_libraries( my_refind
    ${THIRD_PARTY_LIBS}
)
