set (_SRCS

    )

set (_HDRS
    OrthoTree.h
    OrthoTreeParticle.h
    common.h
    BoundingBox.h
    OrthoTreeNode.h
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/OrthoTreeRefactor)
