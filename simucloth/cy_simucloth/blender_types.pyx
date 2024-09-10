from libc.stdint cimport uint32_t, uint64_t

cdef struct AnimData:
    pass

cdef struct Key:
    pass

cdef struct Material:
    pass

cdef struct AssetMetaData:
    pass

cdef struct Library:
    pass

cdef struct IDProperty:
    pass

cdef struct IDOverrideLibrary:
    pass

cdef struct LibraryWeakReference:
    pass

cdef struct BLI_mempool:
    pass

cdef struct CustomDataExternal:
    pass

cdef struct AnonymousAttributeIDHandle:
    pass

cdef struct ImplicitSharingInfoHandle:
    pass

cdef struct ID_Runtime_Remap:
    int status
    int skipped_refcounted
    int skipped_direct
    int skipped_indirect

cdef struct ID_Runtime:
    ID_Runtime_Remap remap

cdef struct ID:
    void *next
    void *prev
    ID *newid
    Library *lib
    AssetMetaData *asset_data
    char name[66]
    short flag
    int tag
    int us
    int icon_id
    uint32_t recalc
    uint32_t recalc_up_to_undo_push
    uint32_t recalc_after_undo_push
    uint64_t session_uuid
    IDProperty *properties
    IDOverrideLibrary *override_library
    ID *orig_id
    void *py_instance
    LibraryWeakReference *library_weak_reference
    ID_Runtime runtime

cdef struct ListBase:
    void *first
    void *last

cdef struct CustomDataLayer:
    int type
    int offset
    int flag
    int active
    int active_rnd
    int active_clone
    int active_mask
    int uid
    char name[68]
    char _pad1[4]
    void *data
    const AnonymousAttributeIDHandle *anonymous_id
    const ImplicitSharingInfoHandle *sharing_info

cdef struct CustomData:
    CustomDataLayer *layers
    int typemap[52]
    char _pad[4]
    int totlayer
    int maxlayer
    int totsize
    BLI_mempool *pool
    CustomDataExternal *external

cdef struct Mesh:
    ID id
    AnimData *adt
    Key *key
    Material **mat
    int totvert
    int totedge
    int totpoly
    int totloop
    int *poly_offset_indices
    CustomData vdata
    CustomData edata
    CustomData pdata
    CustomData ldata
    ListBase vertex_group_names
    int vertex_group_active_index
    int attributes_active_index
