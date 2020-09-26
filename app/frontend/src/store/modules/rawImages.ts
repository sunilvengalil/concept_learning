import { VuexModule, Module, Mutation, Action } from "vuex-module-decorators";
import { IRawImages  } from "@/interfaces";
import { api } from "@/api";
import { mainStore } from "@/utils/store-accessor";


@Module({ name: "raw_images" })
export default class RawImages extends VuexModule {
  rawImages: IRawImages[] = [];

  @Action
  async getImages() {
    try {
      const response = await api.getRawImages(mainStore.token);
      if (response) {
        console.log(response)
        this.setRawImages(response.data);
      }
    } catch (error) {
      await mainStore.checkApiError(error);
    }
  }

  @Mutation
  setRawImages(payload: IRawImages[]) {
    this.rawImages=payload;
  }
}
