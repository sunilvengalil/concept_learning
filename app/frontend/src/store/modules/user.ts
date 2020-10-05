import { api } from "@/api";
import { VuexModule, Module, Mutation, Action } from "vuex-module-decorators";
// import { IRawImages  } from "@/interfaces";
import { mainStore } from "@/utils/store-accessor";
import { IRawImages } from "@/interfaces";

@Module({ name: "user" })
export default class UserModule extends VuexModule {
  rawImages: IRawImages[] = [];

  @Action
  async getImages() {
    try {
      const response = await api.getRawImages(mainStore.token);
      if (response) {
        this.setRawImages(response.data);
      }
    } catch (error) {
      await mainStore.checkApiError(error);
    }
  }

  @Mutation
  setRawImages(payload) {
    this.rawImages=payload;
  }
}
