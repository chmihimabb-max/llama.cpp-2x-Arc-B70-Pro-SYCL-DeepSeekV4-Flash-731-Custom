#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <stdio.h>

int main() {
    if (zesInit(0) != ZE_RESULT_SUCCESS) { printf("zesInit failed\n"); return 1; }
    uint32_t ndrv = 0;
    zesDriverGet(&ndrv, NULL);
    zes_driver_handle_t drv[8];
    zesDriverGet(&ndrv, drv);
    for (uint32_t d = 0; d < ndrv; d++) {
        uint32_t ndev = 0;
        zesDeviceGet(drv[d], &ndev, NULL);
        zes_device_handle_t dev[16];
        zesDeviceGet(drv[d], &ndev, dev);
        for (uint32_t i = 0; i < ndev; i++) {
            uint32_t nmem = 0;
            zesDeviceEnumMemoryModules(dev[i], &nmem, NULL);
            zes_mem_handle_t mem[8];
            zesDeviceEnumMemoryModules(dev[i], &nmem, mem);
            for (uint32_t m = 0; m < nmem; m++) {
                zes_mem_state_t st = { ZES_STRUCTURE_TYPE_MEM_STATE, NULL };
                zesMemoryGetState(mem[m], &st);
                printf("dev%u [mem%u]: total %.2f GiB, free %.2f GiB, used %.2f GiB\n",
                       i, m,
                       st.size / 1073741824.0, st.free / 1073741824.0,
                       (st.size - st.free) / 1073741824.0);
            }
        }
    }
    return 0;
}
