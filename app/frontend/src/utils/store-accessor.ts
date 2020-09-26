import { Store } from "vuex";
import { getModule } from "vuex-module-decorators";
import MainModule from "@/store/modules/main";
import AdminModule from "@/store/modules/admin";
import RawImages from "@/store/modules/rawImages";

let mainStore: MainModule;
let adminStore: AdminModule;
let rawImagesStore: RawImages;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function initializeStores(store: Store<any>): void {
  mainStore = getModule(MainModule, store);
  adminStore = getModule(AdminModule, store);
  rawImagesStore = getModule(RawImages, store);
}

export const modules = {
  main: MainModule,
  admin: AdminModule,
  rawImages:RawImages
};

export { initializeStores, mainStore, adminStore, rawImagesStore };
