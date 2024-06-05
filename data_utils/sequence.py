from transformers import AutoTokenizer, OPTModel
import torch
import numpy as np
import fire


SAT_IMAGE_TOKEN = "[SAT]"
GOAL_IMAGE_TOKEN = "[GOAL]"

RIGHT_ACTION_TOKEN = "[RIGHT]"
LEFT_ACTION_TOKEN = "[LEFT]"
UP_ACTION_TOKEN = "[UP]"
DOWN_ACTION_TOKEN = "[DOWN]"
STOP_ACTION_TOKEN = "[STOP]"

SPECIAL_TOKEN_DICT = {'additional_special_tokens': [SAT_IMAGE_TOKEN,
                                                    GOAL_IMAGE_TOKEN,
                                                    RIGHT_ACTION_TOKEN,
                                                    LEFT_ACTION_TOKEN,
                                                    UP_ACTION_TOKEN,
                                                    DOWN_ACTION_TOKEN,
                                                    STOP_ACTION_TOKEN]}


class Sequence:
    def __init__(
            self,
            dataset_dict,
            tokenizer=None,
            num_image_tokens=1,
            num_action_tokens=1,
            num_goal_tokens=1,
            num_patches=10):
        self.dataset_dict = dataset_dict
        self.tokenizer = tokenizer
        self.num_patches = num_patches
        self.num_image_tokens = num_image_tokens
        self.num_action_tokens = num_action_tokens
        self.num_goal_tokens = num_goal_tokens
        self.current_token_sequence = []
        self.attention_mask = []
        self.action_sequence = []
        self.patch_sequence = []
        self.embedding_sequence = []

    def add_image_tokens(self):
        N = self.num_image_tokens
        I = self.tokenizer.convert_tokens_to_ids(SAT_IMAGE_TOKEN)
        if N is not None or N > 0:
            self.current_token_sequence.extend([I for i in range(N)])
            self.attention_mask.extend([1 for i in range(N)])

    def add_action_tokens(self, action_type="up"):
        N = self.num_action_tokens
        if action_type == 'up':
            I = self.tokenizer.convert_tokens_to_ids(UP_ACTION_TOKEN)
        elif action_type == 'right':
            I = self.tokenizer.convert_tokens_to_ids(RIGHT_ACTION_TOKEN)
        elif action_type == 'down':
            I = self.tokenizer.convert_tokens_to_ids(DOWN_ACTION_TOKEN)
        elif action_type == 'left':
            I = self.tokenizer.convert_tokens_to_ids(LEFT_ACTION_TOKEN)
        else:
            I = self.tokenizer.convert_tokens_to_ids(STOP_ACTION_TOKEN)
        if N is not None or N > 0:
            self.current_token_sequence.extend([I for i in range(N)])
            self.attention_mask.extend([1 for i in range(N)])

    def add_goal_tokens(self):
        N = self.num_goal_tokens
        I = self.tokenizer.convert_tokens_to_ids(GOAL_IMAGE_TOKEN)
        if N is not None or N > 0:
            self.current_token_sequence.extend([I for i in range(N)])
            self.attention_mask.extend([1 for i in range(N)])

    def init_with_goal_image(self, patch_id):
        if self.tokenizer is not None:
            self.add_goal_tokens()
        self.patch_sequence.append(patch_id)
        self.embedding_sequence.append(self.dataset_dict[patch_id])
    
    def init_with_goal_embed(self, embedding, patch_id=12):
        self.patch_sequence.append(patch_id)
        self.embedding_sequence.append(embedding)

    def update_sequence_with_satellite_image_token(self, patch_id):
        if self.tokenizer is not None:
            self.add_image_tokens()
        self.patch_sequence.append(patch_id)
        self.embedding_sequence.append(self.dataset_dict[patch_id])

    def update_sequence_with_action(self, action_type="up"):
        if self.tokenizer is not None:
            self.add_action_tokens(action_type)
            self.add_image_tokens()
        self.action_sequence.append(action_type)
        next_patch_id = -1
        if action_type == "up":
            if self.patch_sequence[-1] not in np.arange(0, self.num_patches, 1):
                next_patch_id = self.patch_sequence[-1] - self.num_patches
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
        elif action_type == "right":
            if self.patch_sequence[-1] not in np.arange(self.num_patches-1, self.num_patches**2, self.num_patches):
                next_patch_id = self.patch_sequence[-1] + 1
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
        elif action_type == "down":
            if self.patch_sequence[-1] not in np.arange(self.num_patches**2-self.num_patches, self.num_patches**2, 1):
                next_patch_id = self.patch_sequence[-1] + self.num_patches
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
        elif action_type == "left":
            if self.patch_sequence[-1] not in np.arange(0, self.num_patches**2, self.num_patches):
                next_patch_id = self.patch_sequence[-1] - 1
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
                self.embedding_sequence.append(
                    self.dataset_dict[self.patch_sequence[-1]])
        
        else:
            next_patch_id = self.patch_sequence[-1]
            self.patch_sequence.append(next_patch_id)
            self.embedding_sequence.append(
                self.dataset_dict[self.patch_sequence[-1]])

    def get_full_dict(self):
        return {'current_token_sequence': self.current_token_sequence,
                'attention_mask': self.attention_mask,
                'action_sequence': self.action_sequence,
                'patch_sequence': self.patch_sequence,
                'embedding_sequence': self.embedding_sequence}

    def get_input_for_model(self, device='cpu'):
        return {
            'input_ids': torch.LongTensor(
                self.current_token_sequence).unsqueeze(0).to(device),
            'attention_mask': torch.Tensor(
                self.attention_mask).unsqueeze(0).to(device),
            'inputs_embeds': torch.FloatTensor(
                np.array(
                    self.embedding_sequence)).unsqueeze(0).to(device),
            'actions': self.action_sequence,
            'patch_sequence': torch.LongTensor(self.patch_sequence).unsqueeze(0).to(device)}


class SequenceDummy:
    def __init__(self, num_patches=10):
        self.num_patches = num_patches
        self.action_sequence = []
        self.patch_sequence = []

    def init_with_goal_image(self, patch_id):
        self.patch_sequence.append(patch_id)

    def update_sequence_with_satellite_image_token(self, patch_id):
        self.patch_sequence.append(patch_id)

    def update_sequence_with_action(self, action_type="up"):
        self.action_sequence.append(action_type)
        next_patch_id = -1
        if action_type == "up":
            if self.patch_sequence[-1] not in np.arange(0, self.num_patches, 1):
                next_patch_id = self.patch_sequence[-1] - self.num_patches
                self.patch_sequence.append(next_patch_id)
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
        elif action_type == "right":
            if self.patch_sequence[-1] not in np.arange(self.num_patches-1, self.num_patches**2, self.num_patches):
                next_patch_id = self.patch_sequence[-1] + 1
                self.patch_sequence.append(next_patch_id)
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
        elif action_type == "down":
            if self.patch_sequence[-1] not in np.arange(self.num_patches**2-self.num_patches, self.num_patches**2, 1):
                next_patch_id = self.patch_sequence[-1] + self.num_patches
                self.patch_sequence.append(next_patch_id)
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
        elif action_type == "left":
            if self.patch_sequence[-1] not in np.arange(0, self.num_patches**2, self.num_patches):
                next_patch_id = self.patch_sequence[-1] - 1
                self.patch_sequence.append(next_patch_id)
            else:
                next_patch_id = self.patch_sequence[-1]
                self.patch_sequence.append(next_patch_id)
        
        else:
            next_patch_id = self.patch_sequence[-1]
            self.patch_sequence.append(next_patch_id)

    def get_full_dict(self):
        return {'action_sequence': self.action_sequence,
                'patch_sequence': self.patch_sequence,}


def test_run(data_path, CURRENT_PATCH=12, GOAL_PATCH=67, tokenizer=None):

    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    dataset_dict = np.load(data_path)
    seq = Sequence(dataset_dict, tokenizer)

    print("#### STARTING SEARCH ####\n\n")

    print(f"CURRENT_PATCH: {CURRENT_PATCH}, GOAL_PATCH: {GOAL_PATCH}\n")

    seq.init_with_goal_image(GOAL_PATCH)
    seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)

    print(f"CURRENT_PATCH_SEQUENCE: {seq.patch_sequence}")
    print(f"CURRENT_TOKEN_SEQUENCE: {seq.current_token_sequence}")
    print(f"CURRENT_ACTION_SEQUENCE: {seq.action_sequence}\n\n")

    print("#### TAKING RANDOM ACTION (only for testing) ####\n\n")

    for i in range(10):
        idx = np.random.randint(0, 5)

        if idx == 0:
            print("#### TAKING ACTION UP ####\n\n")
            seq.update_sequence_with_action("up")
        elif idx == 1:
            print("#### TAKING ACTION RIGHT ####\n\n")
            seq.update_sequence_with_action("right")
        elif idx == 2:
            print("#### TAKING ACTION DOWN ####\n\n")
            seq.update_sequence_with_action("down")
        elif idx == 3:
            print("#### TAKING ACTION LEFT ####\n\n")
            seq.update_sequence_with_action("left")
        else:
            print("#### TAKING ACTION STOP ####\n\n")
            seq.update_sequence_with_action("stop")

        print(f"CURRENT_PATCH_SEQUENCE: {seq.patch_sequence}")
        print(f"CURRENT_TOKEN_SEQUENCE: {seq.current_token_sequence}")
        print(f"CURRENT_ACTION_SEQUENCE: {seq.action_sequence}\n\n")

        if idx==4:
            break

if __name__ == '__main__':
    fire.Fire(test_run)
