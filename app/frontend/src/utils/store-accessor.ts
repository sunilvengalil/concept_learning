import { Store } from "vuex";
import { getModule } from "vuex-module-decorators";
import MainModule from "@/store/modules/main";
import AdminModule from "@/store/modules/admin";
import UserModule from "@/store/modules/user";

let mainStore: MainModule;
let adminStore: AdminModule;
let userStore: UserModule;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function initializeStores(store: Store<any>): void {
  mainStore = getModule(MainModule, store);
  adminStore = getModule(AdminModule, store);
  userStore = getModule(UserModule, store);
}

export const modules = {
  main: MainModule,
  admin: AdminModule,
  user:UserModule
};

export { initializeStores, mainStore, adminStore, userStore };
