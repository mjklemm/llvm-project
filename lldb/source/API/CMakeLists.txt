get_property(LLDB_ALL_PLUGINS GLOBAL PROPERTY LLDB_PLUGINS)

if(LLDB_BUILD_FRAMEWORK)
  set(option_install_prefix INSTALL_PREFIX ${LLDB_FRAMEWORK_INSTALL_DIR})
  set(option_framework FRAMEWORK)
endif()

if(LLDB_ENABLE_PYTHON)
  if (APPLE AND LLVM_ENABLE_LOCAL_SUBMODULE_VISIBILITY)
    # Work around an issue with the Python headers, which have a modular include
    # inside an extern "C" block.
    remove_module_flags()
  endif()
  get_target_property(python_bindings_dir swig_wrapper_python BINARY_DIR)
  set(lldb_python_wrapper ${python_bindings_dir}/LLDBWrapPython.cpp)
endif()

if(LLDB_ENABLE_LUA)
  get_target_property(lua_bindings_dir swig_wrapper_lua BINARY_DIR)
  set(lldb_lua_wrapper ${lua_bindings_dir}/LLDBWrapLua.cpp)
endif()

# Generate SBLanguages.h from Dwarf.def.
set(sb_languages_file
  ${CMAKE_CURRENT_BINARY_DIR}/../../include/lldb/API/SBLanguages.h)
set(sb_languages_generator
  ${LLDB_SOURCE_DIR}/scripts/generate-sbapi-dwarf-enum.py)
add_custom_command(
  COMMENT "Generating SBLanguages.h from Dwarf.def"
  COMMAND "${Python3_EXECUTABLE}"
      ${sb_languages_generator}
      ${LLVM_MAIN_INCLUDE_DIR}/llvm/BinaryFormat/Dwarf.def
      -o ${sb_languages_file}
  OUTPUT ${sb_languages_file}
  DEPENDS ${LLVM_MAIN_INCLUDE_DIR}/llvm/BinaryFormat/Dwarf.def
          ${sb_languages_generator}
  WORKING_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
)
add_custom_target(lldb-sbapi-dwarf-enums
  DEPENDS ${sb_languages_file})
set_target_properties(lldb-sbapi-dwarf-enums PROPERTIES FOLDER "LLDB/Tablegenning")

add_lldb_library(liblldb SHARED ${option_framework}
  SBAddress.cpp
  SBAddressRange.cpp
  SBAddressRangeList.cpp
  SBAttachInfo.cpp
  SBBlock.cpp
  SBBreakpoint.cpp
  SBBreakpointLocation.cpp
  SBBreakpointName.cpp
  SBBreakpointOptionCommon.cpp
  SBBroadcaster.cpp
  SBCommandInterpreter.cpp
  SBCommandInterpreterRunOptions.cpp
  SBCommandReturnObject.cpp
  SBCommunication.cpp
  SBCompileUnit.cpp
  SBData.cpp
  SBDebugger.cpp
  SBDeclaration.cpp
  SBEnvironment.cpp
  SBError.cpp
  SBEvent.cpp
  SBExecutionContext.cpp
  SBExpressionOptions.cpp
  SBFile.cpp
  SBFileSpec.cpp
  SBFileSpecList.cpp
  SBFormat.cpp
  SBFrame.cpp
  SBFunction.cpp
  SBHostOS.cpp
  SBInstruction.cpp
  SBInstructionList.cpp
  SBLanguageRuntime.cpp
  SBLaunchInfo.cpp
  SBLineEntry.cpp
  SBListener.cpp
  SBMemoryRegionInfo.cpp
  SBMemoryRegionInfoList.cpp
  SBModule.cpp
  SBModuleSpec.cpp
  SBMutex.cpp
  SBPlatform.cpp
  SBProcess.cpp
  SBProcessInfo.cpp
  SBProcessInfoList.cpp
  SBProgress.cpp
  SBQueue.cpp
  SBQueueItem.cpp
  SBReproducer.cpp
  SBSaveCoreOptions.cpp
  SBScriptObject.cpp
  SBSection.cpp
  SBSourceManager.cpp
  SBStatisticsOptions.cpp
  SBStream.cpp
  SBStringList.cpp
  SBStructuredData.cpp
  SBSymbol.cpp
  SBSymbolContext.cpp
  SBSymbolContextList.cpp
  SBTarget.cpp
  SBThread.cpp
  SBThreadCollection.cpp
  SBThreadPlan.cpp
  SBTrace.cpp
  SBTraceCursor.cpp
  SBType.cpp
  SBTypeCategory.cpp
  SBTypeEnumMember.cpp
  SBTypeFilter.cpp
  SBTypeFormat.cpp
  SBTypeNameSpecifier.cpp
  SBTypeSummary.cpp
  SBTypeSynthetic.cpp
  SBUnixSignals.cpp
  SBValue.cpp
  SBValueList.cpp
  SBVariablesOptions.cpp
  SBWatchpoint.cpp
  SBWatchpointOptions.cpp
  SystemInitializerFull.cpp
  ${lldb_python_wrapper}
  ${lldb_lua_wrapper}

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/API
  DEPENDS
    lldb-sbapi-dwarf-enums

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInitialization
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbValueObject
    lldbVersion
    ${LLDB_ALL_PLUGINS}

  ${option_install_prefix}
)

# lib/pythonX.Y/dist-packages/lldb/_lldb.so is a symlink to lib/liblldb.so,
# which depends on lib/libLLVM*.so (BUILD_SHARED_LIBS) or lib/libLLVM-10git.so
# (LLVM_LINK_LLVM_DYLIB). Add an additional rpath $ORIGIN/../../../../lib so
# that _lldb.so can be loaded from Python.
if(LLDB_ENABLE_PYTHON AND (BUILD_SHARED_LIBS OR LLVM_LINK_LLVM_DYLIB) AND UNIX AND NOT APPLE)
  set_property(TARGET liblldb APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/../../../../lib${LLVM_LIBDIR_SUFFIX}")
endif()

if(Python3_RPATH)
  set_property(TARGET liblldb APPEND PROPERTY INSTALL_RPATH "${Python3_RPATH}")
  set_property(TARGET liblldb APPEND PROPERTY BUILD_RPATH   "${Python3_RPATH}")
endif()


if(LLDB_ENABLE_PYTHON)
  add_dependencies(liblldb swig_wrapper_python)

  if (MSVC)
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
  else()
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
  endif()

  set_source_files_properties(${lldb_python_wrapper} PROPERTIES GENERATED ON)
  if (CLANG_CL)
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -Wno-unused-function")
  endif()
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -Wno-sequence-point -Wno-cast-qual")
  endif ()
endif()

if(LLDB_ENABLE_LUA)
  add_dependencies(liblldb swig_wrapper_lua)
  target_include_directories(liblldb PRIVATE ${LUA_INCLUDE_DIR})

  if (MSVC)
    set_property(SOURCE ${lldb_lua_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
  else()
    set_property(SOURCE ${lldb_lua_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
  endif()

  set_source_files_properties(${lldb_lua_wrapper} PROPERTIES GENERATED ON)
endif()

set_target_properties(liblldb
  PROPERTIES
  VERSION ${LLDB_VERSION}
)

target_compile_definitions(liblldb PRIVATE LLDB_IN_LIBLLDB)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  if (NOT LLDB_EXPORT_ALL_SYMBOLS)
    # If we're not exporting all symbols, we'll want to explicitly set
    # the exported symbols here.  This prevents 'log enable --stack ...'
    # from working on some systems but limits the liblldb size.
    message(STATUS "Symbols (liblldb): exporting all symbols from the lldb namespace")
    add_llvm_symbol_exports(liblldb ${CMAKE_CURRENT_SOURCE_DIR}/liblldb.exports)
  elseif (NOT LLDB_EXPORT_ALL_SYMBOLS_EXPORTS_FILE)
    # Don't use an explicit export. Instead, tell the linker to export all symbols.
    message(STATUS "Symbols (liblldb): exporting all symbols from the lldb and lldb_private namespaces")
    message(WARNING "Private LLDB symbols frequently change and no API stability is guaranteed. "
                    "Only the SB API is guaranteed to be stable.")
    add_llvm_symbol_exports(liblldb ${CMAKE_CURRENT_SOURCE_DIR}/liblldb-private.exports)
  else ()
    message(STATUS "Symbols (liblldb): exporting all symbols specified in the exports "
            " file '${LLDB_EXPORT_ALL_SYMBOLS_EXPORTS_FILE}'")
    message(WARNING "Private LLDB symbols frequently change and no API stability is guaranteed. "
                    "Only the SB API is guaranteed to be stable.")
    add_llvm_symbol_exports(liblldb "${LLDB_EXPORT_ALL_SYMBOLS_EXPORTS_FILE}")
  endif()
elseif (LLDB_EXPORT_ALL_SYMBOLS)
  message(STATUS "Symbols (liblldb): exporting all symbols from the lldb and lldb_private namespaces")

  # Pull out the various lldb libraries linked into liblldb, these will be used
  # when looking for symbols to extract. We ignore most plugin libraries here,
  # because we may expose more symbols than the DLL limit and these symbols
  # aren't useful to expose.
  get_target_property(all_liblldb_libs liblldb LINK_LIBRARIES)
  set(lldb_libs "")
  foreach(lib ${all_liblldb_libs})
    if(TARGET ${lib} AND ${lib} MATCHES "^lldb" AND
       (${lib} IN_LIST LLDB_EXPORT_ALL_SYMBOLS_PLUGINS OR
        NOT ${lib} MATCHES "^lldbPlugin"))
      get_target_property(lib_type ${lib} TYPE)
      if("${lib_type}" STREQUAL "STATIC_LIBRARY")
        list(APPEND lldb_libs ${lib})
      endif()
    endif()
  endforeach(lib)
  list(REMOVE_DUPLICATES lldb_libs)

  # Extract all of the private symbols and produce a single file we can use for
  # the exports.
  set(exported_symbol_file ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/liblldb_private.symbols)
  get_host_tool_path(llvm-nm LLVM_NM llvm_nm_exe llvm_nm_target)
  add_custom_command(
    OUTPUT ${exported_symbol_file}
    COMMAND "${Python3_EXECUTABLE}"
      ${LLDB_SOURCE_DIR}/scripts/msvc_extract_private_symbols.py
      ${lldb_libs} -o ${exported_symbol_file} --nm=${llvm_nm_exe}
    WORKING_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
    DEPENDS ${LLDB_SOURCE_DIR}/scripts/msvc_extract_private_symbols.py
      ${lldb_libs} ${llvm_nm_target}
    VERBATIM
    COMMENT "Generating liblldb private export list"
  )

  add_llvm_symbol_exports(liblldb ${exported_symbol_file})
endif()

if (NOT MSVC)
  set_target_properties(liblldb
    PROPERTIES
    OUTPUT_NAME lldb
  )
endif()

# The Clang expression parser in LLDB requires the Clang resource directory to function.
if (TARGET clang-resource-headers)
  # If building alongside Clang, just add a dependency to ensure it is build together with liblldb.
  add_dependencies(liblldb clang-resource-headers)
else()
  # In a standalone build create a symlink from the LLDB library directory that points to the
  # resource directory in the Clang library directory. LLDB searches relative to its install path,
  # and the symlink is created in the same relative path as the resource directory of Clang when
  # building alongside Clang.
  # When building the LLDB framework, this isn't necessary as there we copy everything we need into
  # the framework (including the Clang resourece directory).
  if(NOT LLDB_BUILD_FRAMEWORK)
    set(LLDB_CLANG_RESOURCE_DIR "$<TARGET_FILE_DIR:liblldb>/clang")
    add_custom_command(TARGET liblldb POST_BUILD
      COMMENT "Linking Clang resource dir into LLDB build directory: ${LLDB_CLANG_RESOURCE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
      "${LLDB_EXTERNAL_CLANG_RESOURCE_DIR}" "${LLDB_CLANG_RESOURCE_DIR}"
    )
  endif()
endif()

# Stage all headers in the include directory in the build dir.
file(GLOB public_headers ${LLDB_SOURCE_DIR}/include/lldb/API/*.h)
set(lldb_header_staging_dir ${CMAKE_BINARY_DIR}/include/lldb)
set(generated_public_headers ${LLDB_OBJ_DIR}/include/lldb/API/SBLanguages.h)
file(GLOB root_public_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-*.h)
file(GLOB root_private_headers ${LLDB_SOURCE_DIR}/include/lldb/lldb-private*.h)
list(REMOVE_ITEM root_public_headers ${root_private_headers})

find_program(unifdef_EXECUTABLE unifdef)

add_custom_target(liblldb-header-staging)

if (LLDB_BUILD_FRAMEWORK)
  add_custom_target(lldb-framework-fixup-all-headers)
  add_dependencies(lldb-framework-fixup-all-headers liblldb-header-staging)
  add_dependencies(liblldb lldb-framework-fixup-all-headers)
endif()

foreach(header
    ${public_headers}
    ${generated_public_headers}
    ${root_public_headers})
  get_filename_component(basename ${header} NAME)
  set(staged_header ${lldb_header_staging_dir}/${basename})

  if(unifdef_EXECUTABLE)
    # unifdef returns 0 when the file is unchanged and 1 if something was changed.
    # That means if we successfully remove SWIG code, the build system believes
    # that the command has failed and stops. This is undesirable.
    set(copy_command ${unifdef_EXECUTABLE} -USWIG ${header} > ${staged_header} || (exit 0))
  else()
    set(copy_command ${CMAKE_COMMAND} -E copy ${header} ${staged_header})
  endif()

  add_custom_target(liblldb-stage-header-${basename} DEPENDS ${staged_header})
  add_dependencies(liblldb-stage-header-${basename} lldb-sbapi-dwarf-enums)
  add_dependencies(liblldb-header-staging liblldb-stage-header-${basename})
  add_custom_command(
    DEPENDS ${header} OUTPUT ${staged_header}
    COMMAND ${copy_command}
    COMMENT "LLDB headers: stage LLDB headers in include directory")

  list(APPEND lldb_staged_headers ${staged_header})

  if (LLDB_BUILD_FRAMEWORK)
    set(output_header $<TARGET_FILE_DIR:liblldb>/Headers/${basename})

    add_custom_target(lldb-framework-fixup-header-${basename} DEPENDS ${staged_header})
    add_dependencies(lldb-framework-fixup-header-${basename} liblldb-stage-header-${basename})
    add_dependencies(lldb-framework-fixup-all-headers lldb-framework-fixup-header-${basename})

    add_custom_command(TARGET lldb-framework-fixup-header-${basename} POST_BUILD
      COMMAND "${Python3_EXECUTABLE}" ${LLDB_SOURCE_DIR}/scripts/framework-header-fix.py -f lldb_main -i ${staged_header} -o ${output_header}
      COMMENT "LLDB.framework: Fix up and copy framework headers"
    )
  endif()
endforeach()

add_custom_command(TARGET liblldb-header-staging POST_BUILD
  COMMAND "${Python3_EXECUTABLE}" ${LLDB_SOURCE_DIR}/scripts/version-header-fix.py -i ${LLDB_SOURCE_DIR}/include/lldb/lldb-defines.h -o ${lldb_header_staging_dir}/lldb-defines.h -m ${LLDB_VERSION_MAJOR} -n ${LLDB_VERSION_MINOR} -p ${LLDB_VERSION_PATCH}
)
add_dependencies(liblldb liblldb-header-staging)

if(LLDB_BUILD_FRAMEWORK)
  include(LLDBFramework)

  if (TARGET install-liblldb)
    add_dependencies(install-liblldb
      lldb-framework-cleanup)
  endif()

  if (TARGET install-liblldb-stripped)
    add_dependencies(install-liblldb-stripped
      lldb-framework-cleanup)
  endif()
endif()
