# 

plt拿到的name是fullpath，并且可能有so.1 so.2的后缀，所以这里用contains
```cpp
bool targetLib(const char* name) {
    return adt::StringRef(name).contain("libcuda.so");
}
```

而符号名则可以精致匹配
```cpp
bool targetSym(const char* name) {
    return adt::StringRef(name) == "__printf_chk"
}
```


# plt
```cpp
int install_hooker(PltTable* pltTable, const hook::HookInstaller& installer) {
    CHECK(installer.isTargetLib, "isTargetLib can't be empty!");
    CHECK(installer.isTargetSymbol, "isTargetSymbol can't be empty!");
    CHECK(installer.newFuncPtr, "new_func_ptr can't be empty!");
    if (!installer.isTargetLib(pltTable->lib_name.c_str())) {
        return -1;
    }
    size_t index = 0;
    while (index < pltTable->rela_plt_cnt) {
        auto plt = pltTable->rela_plt + index++;
        if (ELF64_R_TYPE(plt->r_info) != R_JUMP_SLOT) {
            continue;
        }

        size_t idx = ELF64_R_SYM(plt->r_info);
        idx = pltTable->dynsym[idx].st_name;
        MLOG(HOOK, INFO) << pltTable->symbol_table + idx; //在STRTAB中取到符号名
        if (!installer.isTargetSymbol(pltTable->symbol_table + idx)) {
            continue;
        }
        void* addr =
            reinterpret_cast<void*>(pltTable->base_header_addr + plt->r_offset);
        int prot = get_memory_permission(addr);
        if (prot == 0) {
            return -1;
        }
        if (!(prot & PROT_WRITE)) { //plt表中的地址不能写，且不能转化为可写的权限，则return -1;
            if (mprotect(ALIGN_ADDR(addr), page_size, PROT_READ | PROT_WRITE) !=
                0) {
                return -1;
            }
        }
        hook::OriginalInfo originalInfo;
        originalInfo.libName = pltTable->lib_name.c_str();
        originalInfo.basePtr = pltTable->base_addr;
        originalInfo.relaPtr = pltTable->rela_plt;
        originalInfo.pltTablePtr = reinterpret_cast<void**>(addr);
        originalInfo.oldFuncPtr =
            reinterpret_cast<void*>(*reinterpret_cast<size_t*>(addr));
        auto new_func_ptr = installer.newFuncPtr(originalInfo);
        *reinterpret_cast<size_t*>(addr) =
            reinterpret_cast<size_t>(new_func_ptr);
        MLOG(HOOK, INFO) << "store " << new_func_ptr << " to " << addr
                         << " original value:"
                         << *reinterpret_cast<void**>(addr);
        // we will not recover the address protect
        // TODO: move this to uninstall function
        // if (!(prot & PROT_WRITE)) {
        //     mprotect(ALIGN_ADDR(addr), page_size, prot);
        // }
        MLOG(HOOK, INFO) << "replace:" << pltTable->symbol_table + idx
                         << " with " << pltTable->symbol_table + idx
                         << " success";
        if (installer.onSuccess) {
            installer.onSuccess();
        }
    }
    return -1;
}

int retrieve_dyn_lib(struct dl_phdr_info* info, size_t info_size, void* table) {
    using VecTable = std::vector<PltTable>;
    auto* vecPltTable = reinterpret_cast<VecTable*>(table);
    PltTable pltTable;
    pltTable.lib_name = info->dlpi_name ? info->dlpi_name : "";
    pltTable.base_header_addr = (char*)info->dlpi_phdr - info_size;
    pltTable.base_addr = reinterpret_cast<const char*>(info->dlpi_addr);
    // pltTable.base_addr = pltTable.base_header_addr;
    ElfW(Dyn*) dyn;
    MLOG(HOOK, INFO) << "install lib name:" << pltTable.lib_name
                     << " dlpi_addr:" << std::hex
                     << reinterpret_cast<void*>(info->dlpi_addr)
                     << " dlpi_phdr:" << std::hex
                     << reinterpret_cast<const void*>(info->dlpi_phdr)
                     << " info_size:" << info_size;
    /* info->dlpi_phdr是一个数组:const ElfW(Phdr) *dlpi_phdr;
        其中ElfW(Phdr)是一个宏，在64位系统下是Elf64_Phdr，定义在/usr/include/elf.h中，表征了segment的各个属性
        typedef struct
        {
        Elf64_Word	p_type;		// Segment type
        Elf64_Word	p_flags;	// Segment flags
        Elf64_Off	p_offset;	// Segment file offset
        Elf64_Addr	p_vaddr;	// Segment virtual address
        Elf64_Addr	p_paddr;	// Segment physical address
        Elf64_Xword	p_filesz;	// Segment size in file
        Elf64_Xword	p_memsz;	// Segment size in memory
        Elf64_Xword	p_align;	// Segment alignment
        } Elf64_Phdr;
        info->dlpi_phnum;是一个ElfW(Half)类型的变量，表示dlpi_phdr数组的长度，也就是segment的个数
    */

    for (size_t header_index = 0; header_index < info->dlpi_phnum;
         header_index++) {
        if (info->dlpi_phdr[header_index].p_type == PT_DYNAMIC) {
            /*
            info->dlpi_addr: 共享对象的虚拟内存起始地址，相对于进程的地址空间。对于可执行文件，这通常是0
            info->dlpi_phdr[header_index].p_vaddr 第header_index个segment的虚拟起始地址，相对于info->dlpi_addr

            那么二者相加，就是segment的虚拟地址，也就是segment在进程中的实际地址

            ElfW(Dyn)会宏扩展为，定义在/usr/include/elf.h中
            typedef struct
            {
            Elf32_Sword	d_tag;			// Dynamic entry type
            union
                {
                Elf32_Word d_val;			// Integer value
                Elf32_Addr d_ptr;			// Address value
                } d_un;
            } Elf32_Dyn;
            */
            dyn = (ElfW(Dyn)*)(info->dlpi_addr +
                               info->dlpi_phdr[header_index].p_vaddr);
            while (dyn->d_tag != DT_NULL) {
                switch (dyn->d_tag) {
                    /*
                        The address of the string table. Symbol names, dependency names, and other strings required by the runtime linker reside in this table.
                        使用符号表通过offset访问
                    */
                    case DT_STRTAB: {
                        pltTable.symbol_table =
                            reinterpret_cast<char*>(dyn->d_un.d_ptr);
                    } break;
                    case DT_STRSZ: {
                    } break;
                    /*
                        The address of the symbol table.
                    */
                    case DT_SYMTAB: {
                        pltTable.dynsym =
                            reinterpret_cast<ElfW(Sym)*>(dyn->d_un.d_ptr);
                    } break;
                    /*
                        与plt表有关的重定位条目
                        The address of relocation entries associated solely with the procedure linkage table.
                        Separating these relocation entries enables the runtime linker to ignore them 
                        when the object is loaded if lazy binding is enabled. 
                        This element requires the DT_PLTRELSZ and DT_PLTREL elements also be present.
                    */
                    case DT_JMPREL: {
                        pltTable.rela_plt =
                            reinterpret_cast<ElfW(Rela)*>(dyn->d_un.d_ptr);
                    } break;
                    case DT_PLTRELSZ: {
                        pltTable.rela_plt_cnt =
                            dyn->d_un.d_val / sizeof(Elf_Plt_Rel);
                    } break;
                    /*
                        The address of a relocation table. 重定位表
                        PLT_DT_REL 就是 DT_RELA
                    */
                    case PLT_DT_REL: {
                        pltTable.rela_dyn =
                            reinterpret_cast<ElfW(Rel)*>(dyn->d_un.d_ptr);
                    } break;
                    case PLT_DT_RELSZ: {
                        // pltTable->rela_plt_cnt = dyn->d_un.d_val /
                        // sizeof(Elf_Plt_Rel);
                    } break;
                }
                dyn++;
            }
        }
    }
    if (pltTable) {
        vecPltTable->emplace_back(pltTable);
    }
    return 0;
}

```