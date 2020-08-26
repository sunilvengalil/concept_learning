import { VuexModule, Module, Mutation } from "vuex-module-decorators";

export interface ImageAttribute {
  character: string;
  probability: boolean;
  clarity: boolean;
  id: number;
}

@Module({ name: "user" })
export default class UserModule extends VuexModule {
  currentImageAttributes: ImageAttribute[] = [];

  get imageAttributes() {
    return () => {
      return { ...this.currentImageAttributes };
    };
  }

  @Mutation
  addAttribute(attribute: ImageAttribute) {
    this.currentImageAttributes.push(attribute);
  }
}
