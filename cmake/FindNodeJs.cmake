

# set(CMAKE_FIND_DEBUG_MODE ON)

# First find the executable
find_program(NodeJs_EXECUTABLE
  NAMES node nodejs
  HINTS $ENV{NODE_DIR}
  PATH_SUFFIXES bin
  DOC "Node.js interpreter"
)

# Find `npm`
find_program(NodeJs_NPM_EXECUTABLE
  NAMES npm
  HINTS $ENV{NODE_DIR}
  PATH_SUFFIXES bin
  DOC "Node Package Manager (NPM) executable"
)

# Find `npx`
find_program(NodeJs_NPX_EXECUTABLE
  NAMES npx
  HINTS $ENV{NODE_DIR}
  PATH_SUFFIXES bin
  DOC "NPM executable runner"
)

find_path(NodeJs_INCLUDE_DIR
  NAMES node.h
  PATH_SUFFIXES node
)

macro(parse_define_number define_name file_string output_variable)
  string(REGEX MATCH "#define ${define_name} ([0-9]+)" _ "${file_string}")
  set(${output_variable} "${CMAKE_MATCH_1}")
endmacro()

if (DEFINED NodeJs_INCLUDE_DIR)

  # message(STATUS "NodeJs_INCLUDE_DIR: ${NodeJs_INCLUDE_DIR}")

  find_file(NodeJs_VERSION_FILE
    NAMES node_version.h
    PATHS ${NodeJs_INCLUDE_DIR}
    NO_DEFAULT_PATH
  )

  if (DEFINED NodeJs_VERSION_FILE)
    # message(STATUS "NodeJs_VERSION_FILE: ${NodeJs_VERSION_FILE}")

    file(READ ${NodeJs_VERSION_FILE} version_file_string)

    parse_define_number(NODE_MODULE_VERSION ${version_file_string} NodeJs_NODE_MODULE_VERSION)
    parse_define_number(NODE_MAJOR_VERSION ${version_file_string} NodeJs_NODE_MAJOR_VERSION)
    parse_define_number(NODE_MINOR_VERSION ${version_file_string} NodeJs_NODE_MINOR_VERSION)
    parse_define_number(NODE_PATCH_VERSION ${version_file_string} NodeJs_NODE_PATCH_VERSION)

    # Set the version variable
    set(NodeJs_VERSION "${NodeJs_NODE_MAJOR_VERSION}.${NodeJs_NODE_MINOR_VERSION}.${NodeJs_NODE_PATCH_VERSION}")

    message(VERBOSE "Detected Node Version ${NodeJs_VERSION}, Module Version: ${NodeJs_NODE_MODULE_VERSION}")

    # With the module version, append this suffix to the search
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NodeJs_NODE_MODULE_VERSION}")
    find_library(NodeJs_LIBRARY
      NAMES node
    )
    list(POP_BACK CMAKE_FIND_LIBRARY_SUFFIXES)
  endif()

endif()

# set(CMAKE_FIND_DEBUG_MODE OFF)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NodeJs
  FOUND_VAR NodeJs_FOUND
  REQUIRED_VARS
    NodeJs_EXECUTABLE
    NodeJs_INCLUDE_DIR
    NodeJs_LIBRARY
    NodeJs_NPM_EXECUTABLE
    NodeJs_NPX_EXECUTABLE
  VERSION_VAR NodeJs_VERSION
)

if(NodeJs_FOUND)
  set(NodeJs_LIBRARIES ${NodeJs_LIBRARY})
  set(NodeJs_INCLUDE_DIRS ${NodeJs_INCLUDE_DIR})
  # set(NodeJs_DEFINITIONS $"")
endif()

if(NodeJs_FOUND AND NOT TARGET NodeJs::Node)
  add_library(NodeJs::Node UNKNOWN IMPORTED)
  set_target_properties(NodeJs::Node PROPERTIES
    IMPORTED_LOCATION "${NodeJs_LIBRARY}"
    # INTERFACE_COMPILE_OPTIONS "${PC_NodeJs_CFLAGS_OTHER}"
    INTERFACE_INCLUDE_DIRECTORIES "${NodeJs_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(
  NodeJs_EXECUTABLE
  NodeJs_INCLUDE_DIR
  NodeJs_LIBRARY
  NodeJs_NPM_EXECUTABLE
  NodeJs_NPX_EXECUTABLE
  NodeJs_VERSION_FILE
)
